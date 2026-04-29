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

class RandomBoardSymmetry:
    def __call__(self, x):
        k = random.randint(0, 7)
        # D₄ group: 8 transformations
        if k == 0:
            return x  # identity
        elif k == 1:
            return x.flip(-1)  # horizontal flip
        elif k == 2:
            return x.flip(-2)  # vertical flip
        elif k == 3:
            return x.transpose(-1, -2)  # transpose
        elif k == 4:
            return x.transpose(-1, -2).flip(-1)  # transpose + H
        elif k == 5:
            return x.transpose(-1, -2).flip(-2)  # transpose + V
        elif k == 6:
            return x.rot90(1, dims=(-2, -1))
        elif k == 7:
            return x.rot90(3, dims=(-2, -1))

train_transform = transforms.Compose([
    RandomBoardSymmetry()
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



class RandomBoardSymmetry_Policy:
    def __call__(self, _):
        k = random.randint(0, 7)

        def transform(t):
            if k == 0:
                return t
            elif k == 1:
                return t.flip(-1)
            elif k == 2:
                return t.flip(-2)
            elif k == 3:
                return t.transpose(-1, -2)
            elif k == 4:
                return t.transpose(-1, -2).flip(-1)
            elif k == 5:
                return t.transpose(-1, -2).flip(-2)
            elif k == 6:
                return t.rot90(1, dims=(-2, -1))
            elif k == 7:
                return t.rot90(3, dims=(-2, -1))
        return transform

train_transform_Policy = transforms.Compose([
    RandomBoardSymmetry_Policy()
])

class CustomTensorDataset_Policy(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            t = self.transform(None)  # get a shared random transform
            x = t(x)
            y = t(y)

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


def train_model_policy(device, model, criterion, loader, nep, optimizer, dtype=torch.float32):
    model.to(device)
    model.train()  # Set the model to training mode
    
    for epoch in range(nep):
        running_loss = 0.0

        for Xb, Yb in tqdm(loader):
            if Xb.size(0) > 1:
                Xb = Xb.to(device, non_blocking=True).to(dtype)
                Yb = Yb.to(device, non_blocking=True)

                # Forward pass
                logits = model(Xb)  # [B, 3, 8, 8]

                # Compute loss across all 3 heads
                loss = 0
                for i in range(3):  # from, to, arrow
                    logits_i = logits[:, i]                    # [B, 8, 8]
                    target_i = Yb[:, i]                        # [B, 8, 8]
                    target_flat = target_i.view(Xb.size(0), -1)#.argmax(dim=1)  # one-hot → label index
                    logits_flat = logits_i.view(Xb.size(0), -1)                # [B, 64]
                    loss += criterion(logits_flat, target_flat)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        # Average loss per epoch
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}/{nep}, Training Loss: {epoch_loss:.4f}")

    return model, epoch_loss

def worker(args):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Qmodel, Policy, bsize, n_game, n_task, temp_args, max_action, randomdir, randomtransform, eval_device, transform_score, dr_noise, prune_chance, seed = args
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.cuda.set_device(eval_device)

    if 'cuda' in eval_device:
        eval_dtype = torch.float16
        Qmodel.to(eval_device).to(eval_dtype)
        Policy.to(eval_device).to(eval_dtype)
        Qmodel = torch.compile(Qmodel)
        Policy = torch.compile(Policy)
    else:
        eval_dtype = torch.float32


    with torch.inference_mode():
        #Qmodel = torch.compile(Qmodel)
        X, Y, S, A, wins, eval_time = selfplay_batch_gpu(Qmodel, Policy, bsize, n_game, n_task, temp_args, max_action, randomdir, randomtransform, eval_device, 
        transform_score, dr_noise, prune_chance)
    
    return X, Y, S, A, wins, eval_time

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
    sp_batch_size = 32

    batch_games = 5000
    boardsize = 8

    temp_args = (0.1, 2.0, 0, 3) # Base, Scale, Power, Policy temperature. follows t = B * ceil( floor(turn // S) + 1) ** P
    max_action = 100 # start in # 3260000
    prune_chance = 0.9 # chance of action pruning.
    randomdir = True
    randomtransform = True

    batch_size = 2048
    #nepoch = 2
    l2_reg_strength = 1e-8
    #lr = 0.000001

    dr_noise = 0.03

    def get_nepoch_lr(current_games):
        if current_games < 100000:
            nepoch = 8
            lr = 0.0003
            update_freq = 5000
        elif current_games < 300000:
            nepoch = 4
            lr = 0.0001
            update_freq = 5000
        elif current_games < 1000000:
            nepoch = 4
            lr = 0.00003
            update_freq = 5000
        elif current_games < 3000000:
            nepoch = 2
            lr = 0.00001
            update_freq = 5000
        else:
            nepoch = 2
            lr = 0.000003
            update_freq = 5000
        return nepoch, lr, update_freq

    m, X, B, c = 4, boardsize, 16, 64  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
    Qmodel = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    model_version = 'v0_3-PG'

    Qmodel_inference = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    model_version = 'v0_3-PG'

    B_policy, c_policy = 8, 64
    Policy_model = PolicyNet_j(m, X, B_policy, c_policy) # small policy network
    Policy_model_inference = PolicyNet_j(m, X, B_policy, c_policy) # small policy network

    transform_score = True

    
    try:
        mnames = os.listdir(os.path.join(wd,'models'))
        mmax = max([int(m[:-4].split('_')[-1]) for m in mnames if f'Pmodel_{model_version}_B{B}C{c}' in m])
        Qmodel.load_state_dict(torch.load(os.path.join(wd,'models',f'Pmodel_{model_version}_B{B}C{c}_{str(mmax).zfill(10)}.pth'),weights_only=True))
        
        current_games = mmax
        try:
            Qmodel.load_state_dict(torch.load(os.path.join(wd,'checkpoint.pth'),weights_only=True))
            print('Loaded a checkpoint')
        except:
            pass
    except:
        print('Initialize new model from 0')
        current_games = 0
        torch.save(Qmodel.state_dict(),os.path.join(wd,'models',f'Pmodel_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        
    print('Current:', current_games)
    
    try:
        Policy_model.load_state_dict(torch.load(os.path.join(wd,'models',f'Policy_{model_version}_B{B_policy}C{c_policy}_{str(mmax).zfill(10)}.pth'),weights_only=True))
        print('Loaded a version for Policy')
        try:
            Policy_model.load_state_dict(torch.load(os.path.join(wd,'checkpoint_Policy.pth'),weights_only=True))
            print('Loaded a checkpoint for Policy')
        except:
            pass
    except:
        print('New policy model')
        except:
            pass
        pass


    pool = Pool(processes=num_processes)

    # initialize compiled model
    print('Recompile')
    Qmodel_inference.load_state_dict(Qmodel.state_dict())
    Qmodel_inference.eval()

    Policy_model_inference.load_state_dict(Policy_model.state_dict())
    Policy_model_inference.eval()
    
    while current_games < 10000000:

        nepoch, lr, update_freq = get_nepoch_lr(current_games)

        t0 = time.time()
        
        n_epsgames = 0
        
        t0 = time.time()

        chunks_ngame = np.diff(np.linspace(0,batch_games,num_processes+1).astype(np.int32))
        print('Assignments:',chunks_ngame)

        # If multiple GPUs, assign to devices
        eval_devices = []
        for i in range(num_processes):
            gpu_id = i % len(cuda_devices)  # Distribute GPUs in a round-robin fashion
            eval_devices.append(f"cuda:{gpu_id}")

        seeds = np.random.randint(-2**32,2**32-1,num_processes)
        
        args = [(Qmodel_inference, Policy_model_inference, boardsize, sp_batch_size, chunks_ngame[_], temp_args, max_action, randomdir, randomtransform, eval_devices[_], transform_score, dr_noise, prune_chance, seeds[_]) for _ in range(num_processes)]
        results = pool.map(worker, args)

        Xd = torch.cat([res[0] for res in results])
        Yd = torch.cat([res[1] for res in results])

        Sd = torch.cat([res[2] for res in results])
        Ad = torch.cat([res[3] for res in results])

        wins = np.sum(np.stack([res[4] for res in results]),axis=0)

        eval_time = sum([res[5] for res in results])

        print(Xd.shape, Yd.shape, Sd.shape, Ad.shape, wins, Yd.mean().item())
        #torch.save(Xd.to(torch.int8),os.path.join(wd,'training',f'X_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        #torch.save(Yd,os.path.join(wd,'training',f'Y_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        #torch.save(Sd.to(torch.int8),os.path.join(wd,'training',f'S_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
        #torch.save(Ad,os.path.join(wd,'training',f'A_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))

        del results

        # Train Policy using current value model
        print('Training Policy', Sd.shape, Ad.shape)

        Sd = Sd.unsqueeze(1)
        Rd = Sd.clone()
        Rd = 1-Rd.sum((-1,-2),keepdims=True)//2%2*2 + torch.zeros_like(Sd)
        Sd = torch.concat([Sd,Rd],dim=1)
        Nacts = Ad.shape[0]
        Adfull = torch.zeros((Nacts, 3, 8, 8), dtype=Ad.dtype)
        rows = Ad[:, [0, 2, 4]].to(torch.long)  # Ensure indices are long
        cols = Ad[:, [1, 3, 5]].to(torch.long)
        batch_indices = torch.arange(Nacts, dtype=torch.long).unsqueeze(1).expand(-1, 3)
        head_indices = torch.tensor([0, 1, 2], dtype=torch.long).expand(Nacts, -1)
        Adfull[batch_indices.reshape(-1), head_indices.reshape(-1), rows.reshape(-1), cols.reshape(-1)] = 1
        
        train_dataset_Policy = CustomTensorDataset_Policy((Sd.float(), Adfull.float()), transform=train_transform_Policy)
        del Rd, Ad, rows, cols, batch_indices, head_indices, Adfull
        
        train_loader_Policy = DataLoader(train_dataset_Policy, batch_size=batch_size, shuffle=True, pin_memory=True)
        opt = torch.optim.Adam(Policy_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2_reg_strength)
        Policy_model, loss = train_model_policy('cuda', Policy_model, torch.nn.CrossEntropyLoss(), train_loader_Policy, nepoch//2, opt)
        Policy_model.to('cpu')
        del train_dataset_Policy, train_loader_Policy
        torch.cuda.empty_cache()
        gc.collect()

        # Train Value
        print('Training Value')
        train_dataset = CustomTensorDataset((Xd, Yd), transform=train_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        opt = torch.optim.Adam(Qmodel.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2_reg_strength)
        Qmodel, loss = train_model('cuda', Qmodel, torch.nn.MSELoss(), train_loader, nepoch, opt)
        Qmodel.to('cpu')
        del train_dataset, train_loader, Xd, Yd
        torch.cuda.empty_cache()
        gc.collect()

        current_games += batch_games

        if current_games % 20000 == 0:
            print('Save model and restart pool')
            torch.save(Qmodel.state_dict(),os.path.join(wd,'models',f'Pmodel_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))
            torch.save(Policy_model.state_dict(),os.path.join(wd,'models',f'Policy_{model_version}_B{B}C{c}_{str(current_games).zfill(10)}.pth'))

            pool.close()
            pool.join()
            gc.collect()
            pool = Pool(processes=num_processes)
            
        if current_games % update_freq == 0:
            print('Update model weight for selfplay')
            Qmodel_inference.load_state_dict(Qmodel.state_dict())
            Qmodel_inference.eval()

            Policy_model_inference.load_state_dict(Policy_model.state_dict())
            Policy_model_inference.eval()

        torch.save(Qmodel.state_dict(),os.path.join(wd,f'checkpoint.pth'))
        torch.save(Policy_model.state_dict(),os.path.join(wd,f'checkpoint_Policy.pth'))

        t1 = time.time()
        print('Model Params:',m, X, B, c,'\n',
               f'Played games: {current_games}, Wall clock: {round(t1-t0,1)}s.\n' \
              +f'N process: {num_processes}. SP Batch size: {sp_batch_size}. \n' \
              +f'Evaluation time: {round(eval_time,1)}s. Per process: {round(eval_time/num_processes,1)}s.\n')
        #quit()