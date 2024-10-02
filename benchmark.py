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


def worker(args):
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    Qmodel, bsize, n_game, n_task, temp_args, randomdir, eval_device, seed = args

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    with torch.inference_mode():
        X, Y, wins, eval_time = selfplay_batch_gpu(Qmodel, bsize, n_game, n_task, temp_args, randomdir, eval_device)
    
    return X, Y, wins, eval_time

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    wd = os.path.dirname(__file__)


    cuda_devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} (cuda:{i})")
            cuda_devices.append(f'cuda:{i}')


    batch_size = 256
    nepoch = 1
    l2_reg_strength = 1e-6
    lr = 0.00001
    temp_args = (1.0, 2.0, -2.0)
    randomdir = True

    boardsize = 8
    batch_games = 256

    #num_processes = 4
    #sp_batch_size = 8

    nproc_list = [1,2,4,6,8,12,24]
    nproc_list = [4,6,8,12]
    sp_list = [16,8,4,2,1]

    

    for num_processes in nproc_list:
        for sp_batch_size in sp_list:
            if sp_batch_size * num_processes <= 128:

                current_games = 0

                print('Start Benchmark Loop')
                m, X, B, c = 4, boardsize, 6, 64  # m input channels, X*X input size, N residual blocks, c channels
                mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
                Qmodel = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
                model_version = 'v0_1'

                pool = Pool(processes=num_processes)

                t0 = time.time()
                
                n_epsgames = 0

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
                
                args = [(Qmodel, boardsize, sp_batch_size, chunks_ngame[_], temp_args, randomdir, eval_devices[_], seeds[_]) for _ in range(num_processes)]
                results = pool.map(worker, args)

                Xd = torch.cat([res[0] for res in results])
                Yd = torch.cat([res[1] for res in results])
                wins = np.sum(np.stack([res[2] for res in results]),axis=0)

                eval_time = sum([res[3] for res in results])

                print(Xd.shape, Yd.shape, wins)
                #quit()
                del results

                torch.cuda.empty_cache()
                gc.collect()

                current_games += batch_games


                pool.close()
                pool.join()
                gc.collect()

                t1 = time.time()
                print('Model Params:',m, X, B, c,'\n',
                        'Version:', model_version,'\n',
                        f'Played games: {current_games}, Wall clock: {round(t1-t0,1)}s.\n' \
                        +f'N process: {num_processes}. SP Batch size: {sp_batch_size}. \n' \
                        +f'Evaluation time: {round(eval_time,1)}s. Per process: {round(eval_time/num_processes,1)}s.\n')
