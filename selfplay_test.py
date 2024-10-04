from base_utils import *
from model_utils import *
import time
#import torch_tensorrt

Turn_mapper = {1:-1,0:1}


def batch_process_boards(boards, roles, actions_list):
    """
    Process a batch of boards with corresponding roles and actions using advanced indexing.

    Args:
        boards (torch.Tensor): A batch of boards (N x bsize x bsize) where N is the number of boards.
        roles (torch.Tensor): A tensor of roles (N,).
        actions_list (torch.Tensor): A tensor of actions (N x 6), where each row corresponds to an action.

    Returns:
        torch.Tensor: A tensor containing updated boards.
    """
    # Get batch size
    batch_size, bsize, _ = boards.shape
    
    # Create copies of boards to avoid modifying the original input
    updated_boards = boards.clone()

    # Unpack the actions into separate components
    x1, y1, x2, y2, x3, y3 = actions_list[:, 0], actions_list[:, 1], actions_list[:, 2], actions_list[:, 3], actions_list[:, 4], actions_list[:, 5]

    # Update the boards in a vectorized manner
    #print(updated_boards.dtype,roles.dtype)
    updated_boards[torch.arange(batch_size), x1, y1] = 0  # Set old positions to 0
    updated_boards[torch.arange(batch_size), x2, y2] = roles  # Move the role to the new position
    updated_boards[torch.arange(batch_size), x3, y3] = 2  # Mark the shooting target with 2

    # Create a tensor for the role in each board
    role_tensor = torch.zeros((batch_size, bsize, bsize), dtype=boards.dtype) + roles.view(-1, 1, 1)

    # Stack the updated boards with the role tensors
    model_inputs = torch.stack([updated_boards, role_tensor], dim=1)  # Shape (N, 2, bsize, bsize)
    
    return model_inputs


#@profile
def selfplay_batch_gpu(Model, bsize, n_game=10, n_task=100, temp_args=(1.0, 2.0, -1.0), randomdir=False, randomtransform=False, eval_device='cuda'):
    Model.eval()
    
    board_dtype = torch.int8
    if 'cuda' in eval_device:
        eval_dtype = torch.float16
    else:
        eval_dtype = torch.float32

    '''example_input = torch.randn((1, 2, 8, 8),dtype=torch.float16).cuda()

    Model = torch_tensorrt.compile(
    Model,
    ir="torch_compile",
    inputs=[torch_tensorrt.Input(example_input.shape)],
    enabled_precisions={torch.float16}  # FP16 precision (if supported)
    )'''
    
    Game_boards = torch.stack([start_board(bsize,board_dtype,randomdir) for _ in range(n_game)])

    Turns = np.zeros(n_game,dtype=np.int32) # [0 for _ in range(n_game)] # Turn % 2 == 0: white (1) move

    Active = np.array([True for _ in range(n_game)])
    
    Index = torch.arange(0,n_game,dtype=torch.int32)

    data_X = [[] for _ in range(n_game)]
    data_Y = [[] for _ in range(n_game)]

    data_S = [[] for _ in range(n_game)]
    data_A = [[] for _ in range(n_game)]

    n_submit = n_game
    n_finish = 0

    wins = np.zeros(2,dtype=np.int32)

    eval_time = 0

    while True in Active:
        print(f'{n_finish} / {n_submit} / {n_task}',end='\r')

        model_inputs = []
        model_idxs = []
        acts_list = []

        # prepare actions
        #print('---',Turns, end='\r')
        for _ in Index[Active]:
            board = Game_boards[_]
            if randomtransform:
                board = random_rotate_and_flip(board)
            role = Turn_mapper[Turns[_]%2]
            # obtain final states based on avail actions
            actions = np.array(select_action_cpp(board,role),dtype=np.int32)
            #print(actions)
            if len(actions) == 0: # conclude the game!!!
                n_finish += 1
                if role == -1:
                    wins[0] += 1
                else:
                    wins[1] += 1
                # calculate q value, gather selfplay data
                #print(role,'Losses')
                
                # current role loses. backtrack q
                # last q is 1 because last player won.
                q_values = torch.ones(int(Turns[_]),dtype=eval_dtype)

                q_values[1::2] *= 0
                q_values = torch.flip(q_values,dims=(0,))
                #q_values[-2::-2] *= -1

                #print(q_values,role,Turns[_])
                data_Y[_].append(q_values)

                Active[_] = False
                Turns[_] = -1
                # reinitialize
                if n_submit < n_task:
                    n_submit += 1
                    #print('Regen')

                    Active[_] = True
                    Turns[_] = 0
                    
                    board = start_board(bsize,board_dtype,randomdir)
                    Game_boards[_] = board
                    role = Turn_mapper[0]
                    
                    actions = np.array(select_action_cpp(board,role),dtype=np.int32)

                else:
                    # skip this action creation
                    #print('End 0')
                    continue

            #rtensor = torch.zeros((bsize,bsize),dtype=eval_dtype)+role

            roles = torch.zeros(len(actions),dtype=board_dtype) + role

            model_inputs.append(batch_process_boards(board.unsqueeze(0).repeat(len(actions), 1, 1),roles,actions))

            model_idxs.append(len(actions))
            acts_list.append(actions)

            del roles

        if True not in Active:
            #print('End 1')
            break
        #print(model_idxs)

        model_inputs = torch.cat(model_inputs)
        #print(model_inputs.shape)
        t0 = time.time()
        model_outputs = Model(model_inputs.to(eval_device).to(eval_dtype)).to('cpu')
        eval_time += (time.time()-t0)
        #print(model_outputs.shape)
        #quit()

        # get model output
        for iout, _ in enumerate(Index[Active]):

            idx_start = sum(model_idxs[:iout])
            idx_end = sum(model_idxs[:iout+1])

            output = model_outputs[idx_start:idx_end].flatten()

            acts = acts_list[iout]

            if len(output) == 1:
                argq = 0
            else:
                t = temp_args[0] * np.ceil((Turns[_]+temp_args[1]) // temp_args[1])**temp_args[2]
                #print(Turns[_],t)
                if t == 0:
                    argq = torch.argmax(output)
                    #best_act = acts[qa]
                else:
                    # get action using probabilistic approach and temperature
                    probabilities = torch.softmax((output / t).to(torch.float32), dim=0)
                    distribution = torch.distributions.Categorical(probabilities)
                    argq = distribution.sample()
                    #best_act = acts[qa]
            
            # next state is essentially first channel of model input
            #action = acts[argq]

            newstates = model_inputs[idx_start:idx_end][argq]

            data_S[_].append(Game_boards[_].clone())
            data_A[_].append(torch.from_numpy(acts[argq]).to(torch.int8))

            Game_boards[_] = newstates[0]

            data_X[_].append(newstates)

            Turns[_] += 1
        
        del model_inputs, model_outputs
        #gc.collect()
        torch.cuda.empty_cache()

    data_X = torch.cat([torch.stack(dx) for dx in data_X])
    data_Y = torch.cat([torch.cat(dy) for dy in data_Y]).unsqueeze(1)

    data_S = torch.cat([torch.stack(ds) for ds in data_S])
    data_A = torch.cat([torch.stack(da) for da in data_A])
    #print(data_X.shape,data_Y.shape)

    return data_X, data_Y, data_S, data_A, wins, eval_time

#@profile
def compete_batch_gpu(Models, bsize, n_game=10, n_task=100, temp_args=(1.0, 2.0, -1.0), randomdir=False, randomtransform=False, eval_device='cuda'):
    board_dtype = torch.int8
    if 'cuda' in eval_device:
        eval_dtype = torch.float16
        Models[0].to(eval_device).to(eval_dtype)
        Models[1].to(eval_device).to(eval_dtype)
    else:
        eval_dtype = torch.float32
    
    Game_boards = torch.stack([start_board(bsize,board_dtype,randomdir) for _ in range(n_game)])

    Turns = np.zeros(n_game,dtype=np.int32) # [0 for _ in range(n_game)] # Turn % 2 == 0: white (1) move

    MasterTurn = 0

    Active = np.array([True for _ in range(n_game)])
    
    Index = torch.arange(0,n_game,dtype=torch.int32)

    n_submit = n_game
    n_finish = 0

    wins = np.zeros(2,dtype=np.int32)

    while True in Active:
        print(f'{n_finish} / {n_submit} / {n_task}',end='\r')

        model_inputs = []
        model_idxs = []
        acts_list = []

        # prepare actions
        #print('---',Turns, end='\r')
        for _ in Index[Active]:
            board = Game_boards[_]
            if randomtransform:
                board = random_rotate_and_flip(board)
            role = Turn_mapper[Turns[_]%2]
            # obtain final states based on avail actions
            actions = select_action_cpp(board,role)
            #print(actions)
            if len(actions) == 0: # conclude the game!!!
                n_finish += 1
                if role == -1:
                    wins[0] += 1
                else:
                    wins[1] += 1

                Active[_] = False
                Turns[_] = -1
                # reinitialize
                if n_submit < n_task:
                    n_submit += 1
                    #print('Regen')

                    #Active[_] = True
                    Turns[_] = 0
                    
                    board = start_board(bsize,board_dtype,randomdir)
                    Game_boards[_] = board
                    role = Turn_mapper[0]
                    
                    actions = select_action_cpp(board,role)

                else:
                    # skip this action creation
                    #print('End 0')
                    continue

            #rtensor = torch.zeros((bsize,bsize),dtype=eval_dtype)+role

            roles = torch.zeros(len(actions),dtype=board_dtype) + role

            model_inputs.append(batch_process_boards(board.unsqueeze(0).repeat(len(actions), 1, 1),roles,np.array(actions,dtype=np.int32)))

            model_idxs.append(len(actions))
            acts_list.append(actions)

            del roles

        if True not in Active:
            #print('End 1')
            break
        #print(model_idxs)

        model_inputs = torch.cat(model_inputs)
        #print(model_inputs.shape)

        if MasterTurn % 2 == 0:
            model_outputs = Models[0](model_inputs.to(eval_device).to(eval_dtype)).to('cpu')
        else:
            model_outputs = Models[1](model_inputs.to(eval_device).to(eval_dtype)).to('cpu')
        #print(model_outputs.shape)
        #quit()

        # get model output
        for iout, _ in enumerate(Index[Active]):

            idx_start = sum(model_idxs[:iout])
            idx_end = sum(model_idxs[:iout+1])

            output = model_outputs[idx_start:idx_end].flatten()

            acts = acts_list[iout]

            if len(output) == 1:
                argq = 0
            else:
                t = temp_args[0] * np.ceil(Turns[_] // temp_args[1] + 1)**temp_args[2]
                #print(Turns[_],t)
                if t == 0:
                    argq = torch.argmax(output)
                    #best_act = acts[qa]
                else:
                    # get action using probabilistic approach and temperature
                    probabilities = torch.softmax((output / t).to(torch.float32), dim=0)
                    distribution = torch.distributions.Categorical(probabilities)
                    argq = distribution.sample()
                    #best_act = acts[qa]
            
            # next state is essentially first channel of model input
            #action = acts[argq]

            newstates = model_inputs[idx_start:idx_end][argq]

            Game_boards[_] = newstates[0]

            Turns[_] += 1
        
        del model_inputs, model_outputs
        #gc.collect()
        torch.cuda.empty_cache()

        MasterTurn += 1

        if MasterTurn % 2 == 0: # start the game when model 0 is the first player
            for _ in Index:
                if Turns[_] == 0 and not Active[_]:
                    Active[_] = True
                elif Turns[_] == -1: # this game is not re-submitted if turn is -1
                    pass


    return wins



if __name__ == '__main__':

    wd = os.path.dirname(__file__)

    
    '''m, X, B, c = 4, 8, 8, 16  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
    model = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
    #torch.save(model.state_dict(),'/home/mingshiyang/AI-Amazon-DQN/test.pth')
    #quit()
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    with torch.inference_mode():
        result = selfplay_batch_gpu(model,8,10,120,(1.0, 2.0, -1.0),True,True,'cuda')

    print(result[1].mean())
    print(result[2])
    quit()'''

    boardsize = 8

    m, X, B, c = 4, boardsize, 6, 96  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
    Qmodel2 = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
    Qmodel2.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint.pth',weights_only=True))

    #m, X, B, c = 2, 8, 6, 64  # m input channels, X*X input size, N residual blocks, c channels
    #mlp_hidden_sizes = [128,64]  # Sizes of hidden layers in the MLP

    #Qmodel2 = Q_V0_0(m, X, B, c, mlp_hidden_sizes)
    #Qmodel2.load_state_dict(torch.load(os.path.join(wd,'models',f'Qmodel_v0_0_B{B}C{c}_{str(2000000).zfill(10)}.pth'),weights_only=True))

    Qmodel2.eval()
    with torch.inference_mode():
        for i in range(1600000,1600001,100000):
            Qmodel1 = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
            Qmodel1.load_state_dict(torch.load(os.path.join(wd,'models',f'Qmodel_v0_1_B{B}C{c}_{str(i).zfill(10)}.pth'),weights_only=True))
            Qmodel1.eval()
            win1 = compete_batch_gpu([Qmodel1,Qmodel2],8,n_game=8,n_task=128,temp_args=(0,2,-1),randomdir=True,randomtransform=True,eval_device='cuda')
            win2 = compete_batch_gpu([Qmodel2,Qmodel1],8,n_game=8,n_task=128,temp_args=(0,2,-1),randomdir=True,randomtransform=True,eval_device='cuda')

            print(i, win1,win2[::-1],win1+win2[::-1],'                    ')
            gc.collect()
            torch.cuda.empty_cache()