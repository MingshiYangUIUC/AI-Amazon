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


def fill_board_winner(b, role, Qmodel, eval_device='cpu', eval_dtype=torch.float32, verbose=False): # let winner play with itself to fill the board, then output final score as number of additional moves.
    # b: board, role: winner
    # set temperature to 0
    points = 1 # start from 1 as a positive score, 1 means ahead 1 turn (just barely winning)
    
    if verbose:
        render_board(b,3,False,[])
        print('Start filling with at least 1 point')
    while True:

        actions = select_action_cpp(b,role)
        if len(actions)==0:
            break
        roles = torch.zeros(len(actions),dtype=b.dtype) + role
        
        model_inputs = batch_process_boards(b.unsqueeze(0).repeat(len(actions), 1, 1),roles,np.array(actions))
        model_inputs = torch.rot90(model_inputs, random.randint(0,3), [-1,-2])

        output = Qmodel(model_inputs.to(eval_device).to(eval_dtype)).to('cpu').flatten()

        argq = torch.argmax(output)

        actions = np.array(actions)
        sel_act = actions[argq]
        
        b = update_board(b,role,sel_act)

        points += 1
        if verbose:
            print(f'Points accumulated: {points}, {role}: Q = {round(output[argq].item()*100,1)}%')
            render_board(b,3,False,[])

    return points

def S2SR(Sd,ax=1,flip=False):
    Sd = Sd.unsqueeze(ax)
    Rd = Sd.clone()
    Rd = 1-Rd.sum((-1,-2),keepdims=True)//2%2*2 + torch.zeros_like(Sd)
    if flip:
        Sd = torch.concat([Sd,Rd*-1],dim=ax)
    else:
        Sd = torch.concat([Sd,Rd],dim=ax)
    return Sd

def S2R(Sd): # return role given board
    R = 1-Sd.sum()//2%2*2
    return int(R.item())

def get_most_likely(policy, actions, t=1):
    """
    Args:
        policy (torch.Tensor): shape [3, 8, 8], output from policy network
        actions (np.ndarray or Tensor): shape [N, 6], each row is [y0, x0, y1, x1, y2, x2]
    
    Returns:
        torch.Tensor: probabilities for each action, shape [N]
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    
    actions = actions.to(policy.device).long()  # Ensure tensor and on same device
    y = actions[:, [0, 2, 4]]  # shape [N, 3]
    x = actions[:, [1, 3, 5]]  # shape [N, 3]
    heads = torch.tensor([0, 1, 2], device=policy.device).view(1, 3).expand(actions.size(0), 3)

    # Gather logits from policy[head, y, x]
    logits = policy[heads, y, x].sum(dim=1)  # shape [N]

    return torch.softmax((logits / t).to(torch.float32), dim=0)

def get_most_likely_j(policy, actions, t=1):
    """
    Args:
        policy (torch.Tensor): shape [3, 8, 8], model output from one joint head
        actions (np.ndarray or Tensor): shape [N, 6], each row is [y0, x0, y1, x1, y2, x2]

    Returns:
        torch.Tensor: probabilities for each action, shape [N]
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    
    actions = actions.to(policy.device).long()
    N = actions.size(0)

    # Flatten policy planes and apply softmax to get probabilities
    log_policy_probs = torch.log_softmax(policy.view(3, -1), dim=1)  # [3, 64]

    # Get flat indices for each component: from, to, arrow
    idx_from  = actions[:, 0] * 8 + actions[:, 1]  # y0 * 8 + x0
    idx_to    = actions[:, 2] * 8 + actions[:, 3]  # y1 * 8 + x1
    idx_arrow = actions[:, 4] * 8 + actions[:, 5]  # y2 * 8 + x2

    # Gather log-probs and sum them (log(P1 * P2 * P3) = logP1 + logP2 + logP3)
    log_joint = (
        log_policy_probs[0, idx_from] +
        log_policy_probs[1, idx_to] +
        log_policy_probs[2, idx_arrow]
    )

    # Apply temperature in log-space
    scaled_log_joint = log_joint / t

    # Final normalized probabilities
    return torch.softmax(scaled_log_joint.to(torch.float32), dim=0)

#@profile
def selfplay_batch_gpu(Model, Policy=None, bsize=8, n_game=10, n_task=100, temp_args=(1.0, 2.0, -1.0, 3), max_action = 9999, randomdir=False, randomtransform=False, eval_device='cuda',
    transform_score = False, dr_noise = 0.1, prune_chance=0.8):
    # if transform_score, output is number of additional turns can be take for winner, or behind for loser.
    
    Model.eval()
    Policy.eval()
    
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
        
        r = torch.rand(1).item()
        if r < prune_chance:
            prune_action=True
        else:
            prune_action=False
        
        print(f'{n_finish} / {n_submit} / {n_task}',end='\r')

        model_inputs = []
        model_idxs = []
        acts_list = []

        # prepare actions
        #print('---',Turns, end='\r')
        
        # get policy estimate
        if prune_action and Policy is not None and True in Active:
            boards = Game_boards[Index[Active]]
            state = S2SR(boards,1)
            policies = Policy(state.to(eval_device).to(eval_dtype)).to('cpu')

        for __, _ in enumerate(Index[Active]):
            board = Game_boards[_]
            if randomtransform:
                board = random_rotate_and_flip(board)
                Game_boards[_] = board
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
                
                if transform_score: # let winner finish game
                    winner = Turn_mapper[(int(Turns[_])-1)%2]
                    point = fill_board_winner(board.clone(), winner, Model, eval_device, eval_dtype, False)
                    # let q value = 0 be negative point, q value = 1 be positive point
                    q_values = torch.ones(int(Turns[_]),dtype=eval_dtype)
                    q_values[1::2] *= 0
                    q_values = torch.flip(q_values,dims=(0,))
                    q_values = q_values * 2 * point - point
                else:
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

            if prune_action and max_action<actions.shape[0]:
                if Policy is not None: # policy guided action subset
                    policy_probs = get_most_likely(policies[__], actions, temp_args[3])
                    noise = torch.distributions.Dirichlet(torch.full_like(policy_probs, dr_noise, dtype=torch.float32)).sample().to(dtype=policy_probs.dtype)
                    noisy_probs = 0.75 * policy_probs + 0.25 * noise
                    actions = actions[torch.flip(torch.argsort(policy_probs),dims=(0,))[:max_action]]
                else:
                    actions = actions[np.random.choice(actions.shape[0], size=max_action, replace=False)]
            
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
                if transform_score:
                    mean = output.mean()
                    std = output.std()
                    
                    # If std is too small or nan, fallback to no scaling
                    if not torch.isfinite(std) or std < 1e-4:
                        output_sample = output - mean  # just center
                    else:
                        output_sample = (output - mean) / std
                else:
                    output_sample = output

                t = temp_args[0] * np.ceil((Turns[_]+temp_args[1]) // temp_args[1])**temp_args[2]
                #print(Turns[_],t)
                if t == 0:
                    argq = torch.argmax(output)
                    #best_act = acts[qa]
                else:
                    # get action using probabilistic approach and temperature
                    probabilities = torch.softmax((output_sample / t).to(torch.float32), dim=0)
                    distribution = torch.distributions.Categorical(probabilities)
                    argq = distribution.sample()
                    #best_act = acts[qa]
            
            # next state is essentially first channel of model input
            #action = acts[argq]

            newstates = model_inputs[idx_start:idx_end][argq]

            # state and action, only record them if not gone through action pruning
            if not prune_action:
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


def selfplay_batch_gpu_distill(Model, Policy=None, bsize=8, n_game=10, n_task=100, temp_args=(1.0, 2.0, -1.0, 3), max_action = 9999, randomdir=False, randomtransform=False, eval_device='cuda',
    transform_score = False, dr_noise = 0.1, prune_chance=0.8):
    # if transform_score, output is number of additional turns can be take for winner, or behind for loser.
    
    Model.eval()
    Policy.eval()
    
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

    data_X = []
    data_Y = []

    data_S = []
    data_A = []

    n_submit = n_game
    n_finish = 0

    wins = np.zeros(2,dtype=np.int32)

    eval_time = 0

    while True in Active:
        
        r = torch.rand(1).item()
        if r < prune_chance:
            prune_action=True
        else:
            prune_action=False
        
        print(f'{n_finish} / {n_submit} / {n_task}',end='\r')

        model_inputs = []
        model_idxs = []
        acts_list = []

        # prepare actions
        #print('---',Turns, end='\r')
        
        # get policy estimate
        if prune_action and Policy is not None and True in Active:
            boards = Game_boards[Index[Active]]
            state = S2SR(boards,1)
            policies = Policy(state.to(eval_device).to(eval_dtype)).to('cpu')

        for __, _ in enumerate(Index[Active]):
            board = Game_boards[_]
            if randomtransform:
                board = random_rotate_and_flip(board)
                Game_boards[_] = board
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
                
                if transform_score: # let winner finish game
                    winner = Turn_mapper[(int(Turns[_])-1)%2]
                    point = fill_board_winner(board.clone(), winner, Model, eval_device, eval_dtype, False)
                    # let q value = 0 be negative point, q value = 1 be positive point
                    q_values = torch.ones(int(Turns[_]),dtype=eval_dtype)
                    q_values[1::2] *= 0
                    q_values = torch.flip(q_values,dims=(0,))
                    q_values = q_values * 2 * point - point
                else:
                    # calculate q value, gather selfplay data
                    #print(role,'Losses')
                    
                    # current role loses. backtrack q
                    # last q is 1 because last player won.
                    q_values = torch.ones(int(Turns[_]),dtype=eval_dtype)

                    q_values[1::2] *= 0
                    q_values = torch.flip(q_values,dims=(0,))

                
                #q_values[-2::-2] *= -1

                #print(q_values,role,Turns[_])
                #data_Y[_].append(q_values)

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

            if prune_action and max_action<actions.shape[0]:
                if Policy is not None: # policy guided action subset
                    policy_probs = get_most_likely(policies[__], actions, temp_args[3])
                    noise = torch.distributions.Dirichlet(torch.full_like(policy_probs, dr_noise, dtype=torch.float32)).sample().to(dtype=policy_probs.dtype)
                    noisy_probs = 0.75 * policy_probs + 0.25 * noise
                    actions = actions[torch.flip(torch.argsort(policy_probs),dims=(0,))[:max_action]]
                else:
                    actions = actions[np.random.choice(actions.shape[0], size=max_action, replace=False)]
            
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
        data_X.append(model_inputs)
        data_Y.append(model_outputs)

        # get model output
        for iout, _ in enumerate(Index[Active]):

            idx_start = sum(model_idxs[:iout])
            idx_end = sum(model_idxs[:iout+1])

            output = model_outputs[idx_start:idx_end].flatten()

            acts = acts_list[iout]

            if len(output) == 1:
                argq = 0
            else:
                if transform_score:
                    mean = output.mean()
                    std = output.std()
                    
                    # If std is too small or nan, fallback to no scaling
                    if not torch.isfinite(std) or std < 1e-4:
                        output_sample = output - mean  # just center
                    else:
                        output_sample = (output - mean) / std
                else:
                    output_sample = output

                t = temp_args[0] * np.ceil((Turns[_]+temp_args[1]) // temp_args[1])**temp_args[2]
                #print(Turns[_],t)
                if t == 0:
                    argq = torch.argmax(output)
                    #best_act = acts[qa]
                else:
                    # get action using probabilistic approach and temperature
                    probabilities = torch.softmax((output_sample / t).to(torch.float32), dim=0)
                    distribution = torch.distributions.Categorical(probabilities)
                    argq = distribution.sample()
                    #best_act = acts[qa]
            
            # next state is essentially first channel of model input
            #action = acts[argq]

            newstates = model_inputs[idx_start:idx_end][argq]

            # state and action, only record them if not gone through action pruning
            if not prune_action:
                data_S.append(Game_boards[_].clone())
                data_A.append(torch.from_numpy(acts[argq]).to(torch.int8))
                #print(len(data_S), len(data_A))

            Game_boards[_] = newstates[0]

            #data_X[_].append(newstates)

            Turns[_] += 1
        
        del model_inputs, model_outputs
        #gc.collect()
        torch.cuda.empty_cache()

    data_X = torch.cat(data_X)
    data_Y = torch.cat(data_Y)

    data_S = torch.stack(data_S,dim=0)
    data_A = torch.stack(data_A,dim=0)
    #print('Finish,', data_X.shape,data_Y.shape,data_S.shape,data_A.shape)

    return data_X, data_Y, data_S, data_A, wins, eval_time


#@profile
def compete_batch_gpu(Models, bsize, n_game=10, n_task=100, temp_args=(1.0, 2.0, -1.0), max_action = (9999,9999), randomdir=False, randomtransform=False, eval_device='cuda', policy_models =[None,None]):
    board_dtype = torch.int8
    if 'cuda' in eval_device:
        eval_dtype = torch.float16
        Models[0].to(eval_device).to(eval_dtype)
        Models[1].to(eval_device).to(eval_dtype)
        if policy_models[0] is not None:
            policy_models[0] = policy_models[0].to(eval_device).to(eval_dtype)
        if policy_models[1] is not None:
            policy_models[1] = policy_models[1].to(eval_device).to(eval_dtype)
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
        if policy_models[MasterTurn%2] is not None and True in Active:
            boards = Game_boards[Index[Active]]
            state = S2SR(boards,1)
            policies = policy_models[MasterTurn%2](state.to(eval_device).to(eval_dtype)).to('cpu')

        for __, _ in enumerate(Index[Active]):
            board = Game_boards[_]
            if randomtransform:
                board = random_rotate_and_flip(board)
            role = Turn_mapper[Turns[_]%2]
            # obtain final states based on avail actions
            actions = np.array(select_action_cpp(board,role))
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
                    
                    actions = np.array(select_action_cpp(board,role))

                else:
                    # skip this action creation
                    #print('End 0')
                    continue

            #rtensor = torch.zeros((bsize,bsize),dtype=eval_dtype)+role
            if MasterTurn % 2 == 0:
                if max_action[0]<actions.shape[0]:
                    if policy_models[0] is None:
                        actions = actions[np.random.choice(actions.shape[0], size=max_action[0], replace=False)]
                    else:
                        policy_probs = get_most_likely_j(policies[__], actions, 3)
                        #noise = torch.distributions.Dirichlet(torch.full_like(policy_probs, 0.1, dtype=torch.float32)).sample().to(dtype=policy_probs.dtype)
                        actions = actions[torch.flip(torch.argsort(policy_probs),dims=(0,))[:max_action[0]]]
            else:
                if max_action[1]<actions.shape[0]:
                    if policy_models[1] is None:
                        actions = actions[np.random.choice(actions.shape[0], size=max_action[1], replace=False)]
                    else:
                        policy_probs = get_most_likely_j(policies[__], actions, 3)
                        #noise = torch.distributions.Dirichlet(torch.full_like(policy_probs, 0.1, dtype=torch.float32)).sample().to(dtype=policy_probs.dtype)
                        actions = actions[torch.flip(torch.argsort(policy_probs),dims=(0,))[:max_action[1]]]

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
                if torch.min(output) < 0: # scale the estimated points
                    mean = output.mean()
                    std = output.std()
                    
                    # If std is too small or nan, fallback to no scaling
                    if not torch.isfinite(std) or std < 1e-4:
                        output_sample = output - mean  # just center
                    else:
                        output_sample = (output - mean) / std
                else:
                    output_sample = output
                t = temp_args[0] * np.ceil(Turns[_] // temp_args[1] + 1)**temp_args[2]
                #print(Turns[_],t)
                if t == 0:
                    argq = torch.argmax(output)
                    #best_act = acts[qa]
                else:
                    # get action using probabilistic approach and temperature
                    probabilities = torch.softmax((output_sample / t).to(torch.float32), dim=0)
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
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
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

    
    mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
    
    #Qmodel2.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint.pth',weights_only=True))

    #m, X, B, c = 2, 8, 6, 64  # m input channels, X*X input size, N residual blocks, c channels
    #mlp_hidden_sizes = [128,64]  # Sizes of hidden layers in the MLP

    #Qmodel2 = Q_V0_0(m, X, B, c, mlp_hidden_sizes)
    #Qmodel2.load_state_dict(torch.load(os.path.join(wd,'models',f'Qmodel_v0_1_B{B}C{c}_{str(1740000).zfill(10)}.pth'),weights_only=True))
    
    #Qmodel1 = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    #Qmodel1.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint.pth',weights_only=True))
    #Qmodel1.eval()
    m, X, B, c = 4, boardsize, 6, 96  # m input channels, X*X input size, N residual blocks, c channels
    Qmodel2 = Q_V0_1(m, X, B, c, mlp_hidden_sizes)
    Qmodel2.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/data/Amazon_model.pth',weights_only=True)['state_dict'])
    #Qmodel2.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/models/Qmodel_v0_1_B6C96_0000400000.pth',weights_only=True))
    Qmodel2.eval()

    #m, X, B, c = 4, boardsize, 16,64
    #Qmodel2 = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    #Qmodel2.load_state_dict(torch.load(os.path.join(wd,'models',f'Pmodel_v0_3-PG_B{B}C{c}_{str(680000).zfill(10)}.pth'),weights_only=True))
    #Qmodel2.eval()

    maxact1 = 2000
    maxact2 = 2000
    n_matches = 1000
    n_game = 32

    m, X, B, c = 4, boardsize, 24, 64  # m input channels, X*X input size, N residual blocks, c channels
    Qmodel1 = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    Qmodel1.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint.pth',weights_only=True))
    Qmodel1.eval()

    m, X, B, c = 4, boardsize, 24, 64  # m input channels, X*X input size, N residual blocks, c channels
    Qmodel2 = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    Qmodel2.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/models/Pmodel_v0_4-PG_B24C64_0004800000.pth',weights_only=True))
    Qmodel2.eval()

    '''m, X, B, c = 4, boardsize, 24, 32  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [64]
    Qmodel1 = P_V0_1(m, X, B, c, mlp_hidden_sizes)
    Qmodel1.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint_D.pth',weights_only=True))
    Qmodel1.eval()'''

    B_policy, c_policy = 8, 64
    Policy_model = PolicyNet_j(m, X, B_policy, c_policy)
    Policy_model.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint_Policy.pth',weights_only=True))
    Policy_model.eval()

    with torch.inference_mode():
        win1 = compete_batch_gpu([Qmodel1,Qmodel2],8,n_game=n_game,n_task=n_matches//2,temp_args=(0.0,2,-1,3),max_action=(maxact1,maxact2),randomdir=True,randomtransform=True,eval_device='cuda',
            policy_models = [None,None])
        win2 = compete_batch_gpu([Qmodel2,Qmodel1],8,n_game=n_game,n_task=n_matches//2,temp_args=(0.0,2,-1,3),max_action=(maxact2,maxact1),randomdir=True,randomtransform=True,eval_device='cuda',
            policy_models = [None,None])
    
    fullres = win1+win2[::-1] # model1 win, model2 win
    win_interval = winrate_confidence_interval(fullres[0], fullres[1])
    print('checkpoint ',win1,win2[::-1],fullres,round(fullres[0] / n_matches*100,1),np.round(win_interval,3)*100,'                    ')
    gc.collect()
    torch.cuda.empty_cache()
    quit()
    with torch.inference_mode():
        for i in range(4100000,5900001,600000):
            Qmodel1 = P_V0_1(m, X, B, c, mlp_hidden_sizes)
            Qmodel1.load_state_dict(torch.load(os.path.join(wd,'models',f'Pmodel_v0_4-PG_B{B}C{c}_{str(i).zfill(10)}.pth'),weights_only=True))
            Qmodel1.eval()
            #Policy_model = PolicyNet_j(m, X, B_policy, c_policy) # small policy network
            #Policy_model.load_state_dict(torch.load(os.path.join(wd,'models',f'Policy_v0_4-PG_B{B_policy}C{c_policy}_{str(i).zfill(10)}.pth'),weights_only=True))
            #Policy_model.eval()

            #m2, X2, B2, c2 = 4, boardsize, 16,64
            #Qmodel2 = P_V0_1(m2, X2, B2, c2, mlp_hidden_sizes)
            #Qmodel2.load_state_dict(torch.load(os.path.join(wd,'models',f'Pmodel_v0_3-PG_B{B2}C{c2}_{str(i).zfill(10)}.pth'),weights_only=True))
            #Qmodel2.eval()

            win1 = compete_batch_gpu([Qmodel1,Qmodel2],8,n_game=n_game,n_task=n_matches//2,temp_args=(0.0,2,-1),max_action=(maxact1,maxact2),randomdir=True,randomtransform=True,eval_device='cuda',
            policy_models = [None,None])
            win2 = compete_batch_gpu([Qmodel2,Qmodel1],8,n_game=n_game,n_task=n_matches//2,temp_args=(0.0,2,-1),max_action=(maxact2,maxact1),randomdir=True,randomtransform=True,eval_device='cuda',
            policy_models = [None,None])
            
            fullres = win1+win2[::-1] # model1 win, model2 win
            win_interval = winrate_confidence_interval(fullres[0], fullres[1])
            print(str(i).zfill(10), win1,win2[::-1],fullres,round(fullres[0] / n_matches*100,1),np.round(win_interval,3)*100,'                    ')
            gc.collect()
            torch.cuda.empty_cache()