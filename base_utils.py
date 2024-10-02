import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import os
import random
from action_module import select_action_cpp
import gc


def random_rotate_and_flip(board):
    """
    Randomly rotate and flip a square tensor.
    
    Args:
        board (torch.Tensor): A 2D square tensor (bsize x bsize).
        
    Returns:
        torch.Tensor: The transformed tensor.
    """
    # Randomly decide the number of 90-degree rotations (0, 1, 2, or 3)
    num_rotations = torch.randint(0, 4, (1,)).item()

    # Apply the rotations using torch.rot90
    rotated_board = torch.rot90(board, num_rotations, [0, 1])

    # Randomly decide whether to flip horizontally (0 or 1)
    if torch.rand(1).item() > 0.5:
        rotated_board = torch.flip(rotated_board, [1])  # Flip horizontally

    # Randomly decide whether to flip vertically (0 or 1)
    if torch.rand(1).item() > 0.5:
        rotated_board = torch.flip(rotated_board, [0])  # Flip vertically

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
        b = random_rotate_and_flip(b)
    return b

def render_board(b, figsize=3, grid=False, texts = []):
    cmap = ['black', 'gray', 'white', 'red']
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(b, cmap=colors.ListedColormap(cmap),vmin=-1,vmax=2)
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

def update_board(b,role,action):
    b[action[0],action[1]] = 0
    b[action[2],action[3]] = role
    b[action[4],action[5]] = 2
    return b

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