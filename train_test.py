import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

from base_utils import *
from model_utils import *
from selfplay_test import *

import time
from torch.multiprocessing import Pool
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm
import os
import gc
import psutil
from collections import deque 
import torch.multiprocessing as mp
from torchvision.transforms import v2
from torchvision import transforms
import torchvision.transforms.v2.functional as F
import random

# Custom transform to randomly rotate by 0, 90, 180, or 270 degrees
class Random90DegreeRotation:
    def __call__(self, x):
        # Use torch.randint to select a random rotation (0, 90, 180, 270 degrees)
        num_rotations = torch.randint(0, 4, (1,)).item()  # 0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
        #print(f"Applying {90 * num_rotations} degrees rotation")
        return torch.rot90(x, num_rotations, [-2, -1])  # Apply the rotation using torch.rot90


train_transform = transforms.Compose([
    v2.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with a probability of 0.5
    v2.RandomVerticalFlip(p=0.5),    # Random vertical flip with a probability of 0.5
    Random90DegreeRotation(),                # Custom random 90-degree rotation
    #transforms.ToTensor()                    # Convert the image to a tensor
])

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        t = self.tensors[2][index]

        if self.transform:
            # Apply the same transform to both x and t
            state = torch.random.get_rng_state()  # Save the random state
            x_transformed = self.transform(x)
            torch.random.set_rng_state(state)  # Reset the random state for t
            t_transformed = self.transform(t)
            return x_transformed, y, t_transformed
        return x, y, t

    def __len__(self):
        return len(self.tensors[0])

def train_model(device, model, criterion_win, criterion_territory, territory_weight, loader, nep, optimizer, dtype=torch.float32):
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, win_targets, territory_targets in tqdm(loader):
            if inputs.size(0) > 1:
                # Move inputs and targets to device
                inputs = inputs.to(device, non_blocking=True).to(dtype)
                win_targets = win_targets.to(device, non_blocking=True).to(dtype)
                territory_targets = territory_targets.to(device, non_blocking=True).to(dtype)
                
                # Forward pass
                win_outputs, territory_outputs = model(inputs)
                
                # Calculate losses for both outputs
                win_loss = criterion_win(win_outputs, win_targets)
                territory_loss = criterion_territory(territory_outputs, territory_targets)
                
                # Combine the losses (you can apply weights here if needed)
                total_loss = win_loss + territory_loss*territory_weight
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                running_loss += total_loss.item()
        
        # Calculate the average loss per batch over the epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    
    return model, epoch_loss


def worker(args):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Qmodel, bsize, n_game, n_task, temp_args, max_action, randomdir, randomtransform, eval_device, seed = args
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.cuda.set_device(eval_device)

    if 'cuda' in eval_device:
        eval_dtype = torch.float16
        Qmodel.to(eval_device).to(eval_dtype)
    else:
        eval_dtype = torch.float32


    with torch.inference_mode():
        Qmodel = torch.compile(Qmodel)
        X, Y, T, wins, eval_time = selfplay_batch_gpu(Qmodel, bsize, n_game, n_task, temp_args, max_action, randomdir, randomtransform, eval_device)
    
    return X, Y, T, wins, eval_time

if __name__ == '__main__':


    mp.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('high')

    wd = os.path.dirname(__file__)


    cuda_devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} (cuda:{i})")
            cuda_devices.append(f'cuda:{i}')
    #eval_devices = cuda_devices
    # set up directory if not done so
    if not os.path.isdir(os.path.join(wd,'data')):
        os.mkdir(os.path.join(wd,'data'))
    if not os.path.isdir(os.path.join(wd,'testdir')):
        os.mkdir(os.path.join(wd,'testdir'))
    if not os.path.isdir(os.path.join(wd,'models')):
        os.mkdir(os.path.join(wd,'models'))
    if not os.path.isdir(os.path.join(wd,'training')):
        os.mkdir(os.path.join(wd,'training'))

    num_processes = 6
    sp_batch_size = 8

    batch_games = 2500
    boardsize = 8

    temp_args = (0.1, 2.0, -1.0) # Base, Scale, Power. follows t = B * ceil( floor(turn // S) + 1) ** P
    max_action = 9999 # start in # 3260000
    randomdir = True
    randomtransform = True

    batch_size = 2048
    nepoch = 2
    l2_reg_strength = 1e-8
    lr = 0.00001

    territory_weight = 0.01


    m, X, B, c = 5, boardsize, 6, 96  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
    Qmodel = Q_V1_0(m, X, B, c, mlp_hidden_sizes)
    model_version = 'v1_0'

    Qmodel_inference = Q_V1_0(m, X, B, c, mlp_hidden_sizes)
    model_version = 'v1_0'

    '''try:
        mname = [f for f in os.listdir(os.path.join(wd)) if 'checkpoint' in f]
        Qmodel.load_state_dict(torch.load(os.path.join(wd,mname),weights_only=True))
        current_games = int(mname[:-4].split('_')[-1])
    except:'''
    
    try:
        mnames = os.listdir(os.path.join(wd,'models'))
        mmax = max([int(m[:-4].split('_')[-1]) for m in mnames if f'Qmodel_{model_version}_B{B}C{c}' in m])
        Qmodel.load_state_dict(torch.load(os.path.join(wd,'models',f'Qmodel_{model_version}_B{B}C{c}_{str(mmax).zfill(10)}.pth'),weights_only=True))
        
        current_games = mmax
        try:
            Qmodel.load_state_dict(torch.load(os.path.join(wd,'checkpoint.pth'),weights_only=True))
            print('Loaded a checkpoint')
        except:
            pass
    except:
        current_games = 0
        torch.save(Qmodel.state_dict(),os.path.join(wd,'models',f'Qmodel_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))

        
    print('Current:', current_games)
    
    

    pool = Pool(processes=num_processes)

    while current_games < 10000000:
        t0 = time.time()
        
        n_epsgames = 0

        Qmodel_inference.load_state_dict(Qmodel.state_dict())
        Qmodel.eval()

        
        t0 = time.time()

        chunks_ngame = np.diff(np.linspace(0,batch_games,num_processes+1).astype(np.int32))
        print('Assignments:',chunks_ngame)

        # If multiple GPUs, assign to devices
        eval_devices = []
        for i in range(num_processes):
            gpu_id = i % len(cuda_devices)  # Distribute GPUs in a round-robin fashion
            eval_devices.append(f"cuda:{gpu_id}")

        seeds = np.random.randint(-2**32,2**32-1,num_processes)
        
        args = [(Qmodel_inference, boardsize, sp_batch_size, chunks_ngame[_], temp_args, max_action, randomdir, randomtransform, eval_devices[_], seeds[_]) for _ in range(num_processes)]
        results = pool.map(worker, args)

        Xd = torch.cat([res[0] for res in results])
        Yd = torch.cat([res[1] for res in results])
        Td = torch.cat([res[2] for res in results])

        wins = np.sum(np.stack([res[3] for res in results]),axis=0)

        eval_time = sum([res[4] for res in results])

        print(Xd.shape, Yd.shape, Td.shape, wins, Yd.mean().item())
        #torch.save(Xd.to(torch.int8),os.path.join(wd,'training',f'X_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        #torch.save(Yd,os.path.join(wd,'training',f'Y_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))

        #torch.save(Td.to(torch.int8),os.path.join(wd,'training',f'T_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        #quit()
        del results

        train_dataset = CustomTensorDataset((Xd, Yd, Td), transform=train_transform)

        #print(SL_X.nbytes/1024**3)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        opt = torch.optim.Adam(Qmodel.parameters(), lr=lr, weight_decay=l2_reg_strength)
        Qmodel, loss = train_model('cuda', Qmodel, torch.nn.MSELoss(), torch.nn.MSELoss(), territory_weight, train_loader, nepoch, opt)
        Qmodel.to('cpu')
        del train_dataset, train_loader, Xd, Yd
        torch.cuda.empty_cache()
        gc.collect()

        current_games += batch_games

        if current_games % 20000 == 0:
            print('Save model and restart pool')
            torch.save(Qmodel.state_dict(),os.path.join(wd,'models',f'Qmodel_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))

            pool.close()
            pool.join()
            gc.collect()
            pool = Pool(processes=num_processes)
        
        torch.save(Qmodel.state_dict(),os.path.join(wd,f'checkpoint.pth'))

        t1 = time.time()
        print('Model Params:',m, X, B, c,'\n',
               f'Played games: {current_games}, Wall clock: {round(t1-t0,1)}s.\n' \
              +f'N process: {num_processes}. SP Batch size: {sp_batch_size}. \n' \
              +f'Evaluation time: {round(eval_time,1)}s. Per process: {round(eval_time/num_processes,1)}s.\n')
        #quit()