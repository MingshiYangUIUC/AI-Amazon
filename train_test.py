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
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        return F.rotate(x, angle)

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

        if self.transform:
            # Apply transformation on the input (x)
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.tensors[0])

def train_model(device, model, criterion, loader, nep, optimizer,dtype=torch.float32):
    #scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    model.train()  # Set the model to training mode
    for epoch in range(nep):
        running_loss = 0.0
        
        for inputs, targets in tqdm(loader):
            if inputs.size(0) > 1:
                inputs, targets = inputs.to(device, non_blocking=True).to(dtype), targets.to(device, non_blocking=True).to(dtype)
                
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize
                optimizer.zero_grad(set_to_none=True)  # Zero the gradients
                
                running_loss += loss.item()
        
        # Calculate the average loss per batch over the epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{nep}, Training Loss: {epoch_loss:.4f}")
    return model, epoch_loss


def worker(args):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Qmodel, bsize, n_game, n_task, temp_args, randomdir, randomtransform, eval_device, seed = args
    
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
        X, Y, S,A,wins, eval_time = selfplay_batch_gpu(Qmodel, bsize, n_game, n_task, temp_args, randomdir, randomtransform, eval_device)
    
    return X, Y, S,A,wins, eval_time

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
    randomdir = True
    randomtransform = True

    batch_size = 2048
    nepoch = 2
    l2_reg_strength = 1e-8
    lr = 0.00001


    m, X, B, c = 4, boardsize, 6, 96  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
    Qmodel = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
    model_version = 'v0_1'

    Qmodel_inference = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
    model_version = 'v0_1'

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
        
        args = [(Qmodel_inference, boardsize, sp_batch_size, chunks_ngame[_], temp_args, randomdir, randomtransform, eval_devices[_], seeds[_]) for _ in range(num_processes)]
        results = pool.map(worker, args)

        Xd = torch.cat([res[0] for res in results])
        Yd = torch.cat([res[1] for res in results])

        Sd = torch.cat([res[2] for res in results])
        Ad = torch.cat([res[3] for res in results])

        wins = np.sum(np.stack([res[4] for res in results]),axis=0)

        eval_time = sum([res[5] for res in results])

        print(Xd.shape, Yd.shape, Sd.shape, Ad.shape, wins, Yd.mean().item())
        torch.save(Xd.to(torch.int8),os.path.join(wd,'training',f'X_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        torch.save(Yd,os.path.join(wd,'training',f'Y_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))

        torch.save(Sd.to(torch.int8),os.path.join(wd,'training',f'S_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        torch.save(Ad,os.path.join(wd,'training',f'A_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        #quit()
        del results

        train_dataset = CustomTensorDataset((Xd, Yd), transform=train_transform)

        #print(SL_X.nbytes/1024**3)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        opt = torch.optim.Adam(Qmodel.parameters(), lr=lr, weight_decay=l2_reg_strength)
        Qmodel, loss = train_model('cuda', Qmodel, torch.nn.MSELoss(), train_loader, nepoch, opt)
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