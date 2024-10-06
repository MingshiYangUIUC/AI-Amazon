import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import os
import random
from action_module import select_action_cpp
import gc


def random_rotate_and_flip(board):
    # Randomly decide the number of 90-degree rotations (0, 1, 2, or 3)
    num_rotations = torch.randint(0, 4, (1,)).item()
    fh,fv = 0,0

    # Apply the rotations using torch.rot90
    rotated_board = torch.rot90(board, num_rotations, [0, 1])

    # Randomly decide whether to flip horizontally (0 or 1)
    if torch.rand(1).item() > 0.5:
        rotated_board = torch.flip(rotated_board, [1])  # Flip horizontally
        fh = 1

    # Randomly decide whether to flip vertically (0 or 1)
    if torch.rand(1).item() > 0.5:
        rotated_board = torch.flip(rotated_board, [0])  # Flip vertically
        fv = 1

    return rotated_board, (num_rotations,fh,fv) # return rotation dies to be used later

def reset_board_rotation_flip(rotated_board,dirs):
    (num_rotations,fh,fv) = dirs
    
    if fv == 1:
        rotated_board = torch.flip(rotated_board, [0])  # Flip vertically
    
    if fh == 1:
        rotated_board = torch.flip(rotated_board, [1])  # Flip horizontally
    # Apply the rotations using torch.rot90
    rotated_board = torch.rot90(rotated_board, 4-num_rotations, [0, 1])

    return rotated_board

def start_board(size=8,dtype=torch.float32,randomdir=False):
    b = torch.zeros((size,size),dtype=dtype) # empty
    if size == 8:
        b[0,[2,5]] = -1
        b[2,[0,7]] = -1
        b[5,[0,7]] = 1
        b[7,[2,5]] = 1
    elif size == 10:
        b[0,[3,6]] = -1
        b[3,[0,9]] = -1
        b[9,[3,6]] = 1
        b[6,[0,9]] = 1
    if randomdir:
        b = random_rotate_and_flip(b)[0]
    return b

def render_board(b, figsize=3, grid=False, texts = [], territory=False):
    if not territory:
        cmap = ['black', 'gray', 'white', 'red']
        vmin=-1
    else:
        cmap = ['blue', 'black', 'gray', 'white', 'pink']
        vmin=-2
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(b, cmap=colors.ListedColormap(cmap),vmin=vmin,vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for txt in texts:
        ax.text(txt[0],txt[1],txt[2],ha='center',va='center',color=txt[3],fontsize=figsize*3)
    if grid:
        ax.set_xticks(np.arange(-0.5, b.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, b.shape[1], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    plt.show()
    pass

directions = np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1) if (i!=0 or j!=0)],dtype=np.int32)
#print(directions)

#@profile
def select_action(b,role=0):
    # action has format (crd, movecrd, shootcrd)

    bsize = b.shape[0]
    coords = np.array(np.where(b == role)).T
    actions = []

    for crd in coords:

        # helper board for shooting
        b1 = b.clone()
        b1[crd[0],crd[1]] = 0

        # Get move places, for each move places, get shooting places
        movetgt = []

        for dir in directions:
            ncrd = crd.copy()
            ncrd = ncrd + dir

            # Vectorize the move to collect targets along the direction
            while np.all(ncrd >= 0) and np.all(ncrd < bsize) and b[ncrd[0], ncrd[1]] == 0:
                movetgt.append(ncrd.copy())
                ncrd += dir

        # Shooting arrow for all move targets
        for mt in movetgt:

            for dir in directions:
                nmt = mt.copy()
                nmt = nmt + dir

                # Vectorize the shooting process
                while np.all(nmt >= 0) and np.all(nmt < bsize) and b1[nmt[0], nmt[1]] == 0:
                    actions.append(np.concatenate([crd, mt, nmt]))
                    nmt += dir

    return actions

def update_board(b,role,action,territory=False):
    b[action[0],action[1]] = 0
    b[action[2],action[3]] = role
    if not territory:
        b[action[4],action[5]] = 2
    else:
        b[action[4],action[5]] = 2*role
    return b

def batch_process_boards(boards, roles, actions_list, territory=False):
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
    if not territory:
        updated_boards[torch.arange(batch_size), x3, y3] = 2  # Mark the shooting target with 2
    else:
        updated_boards[torch.arange(batch_size), x3, y3] = 2*roles

    # Create a tensor for the role in each board
    role_tensor = torch.zeros((batch_size, bsize, bsize), dtype=boards.dtype) + roles.view(-1, 1, 1)

    # Stack the updated boards with the role tensors
    model_inputs = torch.stack([updated_boards, role_tensor], dim=1)  # Shape (N, 2, bsize, bsize)
    
    return model_inputs


def check_finish(actions):
    return len(actions) == 0

if __name__ == '__main__':
    b = start_board(8)
    actions = select_action_cpp(b,-1)

    print(len(actions))
    
    a = random.choice(actions)
    for a in actions:
        b1 = b.clone()

        b1 = update_board(b1,-1,a)
        if torch.sum(b1.flatten()!= 0) != 9:
            render_board(b1)
    print('Pass')
    render_board(b1)