import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Add skip connection
        out = self.relu(out)
        return out

class Q_V0_0(nn.Module):
    def __init__(self, m, X, N, c, mlp_hidden_sizes):
        super(Q_V0_0, self).__init__()
        self.m = m
        self.X = X
        self.N = N
        self.c = c
        
        # Input convolution to map from m input channels to c channels
        self.input_conv = nn.Conv2d(m, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Stack N residual blocks with c channels
        self.resblocks = nn.Sequential(*[ResidualBlock(c) for _ in range(N)])
        
        # After residual blocks, flatten the output
        self.flatten = nn.Flatten()
        
        # Define MLP layers
        mlp_layers = []
        in_size = c * X * X  # Flattened input size based on c channels
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.ReLU(inplace=True))
            in_size = hidden_size
        
        # Output layer to give a value between 0 and 1
        mlp_layers.append(nn.Linear(in_size, 1))
        mlp_layers.append(nn.Sigmoid())  # To output a value between 0 and 1
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        # Initial conv layer
        x = self.input_conv(x)
        x = self.relu(x)
        
        # Pass through N residual blocks
        x = self.resblocks(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Pass through the MLP layers
        x = self.mlp(x)
        
        return x
    
    def get_model_description(self):
        # Returns a dictionary description of the model
        return {
            'n_resblock': self.N,
            'size': self.X,
            'n_feature': self.m,
            'n_channels': self.c
        }


class ResidualBlock_BN(nn.Module):
    def __init__(self, c):
        super(ResidualBlock_BN, self).__init__()
        # Convolutional layer followed by BatchNorm
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)  # BatchNorm for the first conv layer
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)  # BatchNorm for the second conv layer

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Q_V0_1(nn.Module):
    def __init__(self, m, X, N, c, mlp_hidden_sizes):
        super(Q_V0_1, self).__init__()
        self.m = m
        self.X = X
        self.N = N
        self.c = c
        
        # Input convolution to map from m input channels to c channels
        self.input_conv = nn.Conv2d(m, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Stack N residual blocks with c channels
        self.resblocks = nn.Sequential(*[ResidualBlock_BN(c) for _ in range(N)])
        
        self.output_conv = nn.Conv2d(c, 2, kernel_size=1, padding=0)
        # After residual blocks, flatten the output
        self.flatten = nn.Flatten()
        
        # Define MLP layers
        mlp_layers = []
        in_size = 2 * X * X  # Flattened input size based on c channels
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.ReLU(inplace=True))
            in_size = hidden_size

        self.hiddensize = mlp_hidden_sizes

        # Output layer to give a value between 0 and 1
        mlp_layers.append(nn.Linear(in_size, 1))
        mlp_layers.append(nn.Sigmoid())  # To output a value between 0 and 1
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):

        board_channel = x[:,0]

        one_hot = torch.zeros((x.shape[0], 3, self.X, self.X), device=x.device, dtype=x.dtype)
        one_hot[:, 0, :, :] = (board_channel == -1)  # Channel for -1
        one_hot[:, 1, :, :] = (board_channel == 1)   # Channel for 1
        one_hot[:, 2, :, :] = (torch.abs(board_channel) == 2)   # Channel for 2

        # Extract the active player channel (second channel)
        active_player_channel = x[:, 1, :, :].unsqueeze(1)  # Shape (B, 1, Y, X)

        # Concatenate the one-hot channels with the active player channel
        x = torch.cat([one_hot, active_player_channel], dim=1)  # Shape (B, 4, Y, X)

        #print(x[0])
    
        # Initial conv layer
        x = self.input_conv(x)
        x = self.relu(x)
        
        # Pass through N residual blocks
        x = self.resblocks(x)

        x = self.output_conv(x)
        x = self.relu(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Pass through the MLP layers
        x = self.mlp(x)
        
        return x
    
    def get_model_description(self):
        # Returns a dictionary description of the model
        return {
            'name': 'Q_V0_1',
            'n_resblock': self.N,
            'size': self.X,
            'n_feature': self.m,
            'n_channels': self.c,
            'hidden_sizes': '-'.join([str(hs) for hs in self.hiddensize])
        }


class Q_V1_0(nn.Module): # output both q and expected territory (as sign of final gameboard and 0)
    def __init__(self, m, X, N, c, mlp_hidden_sizes):
        super(Q_V1_0, self).__init__()
        self.m = m
        self.X = X
        self.N = N
        self.c = c
        
        # Input convolution to map from m input channels to c channels
        self.input_conv = nn.Conv2d(m, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Stack N residual blocks with c channels
        self.resblocks = nn.Sequential(*[ResidualBlock_BN(c) for _ in range(N)])
        
        self.flatten = nn.Flatten()

        self.output_conv_v = nn.Conv2d(c, 2, kernel_size=1, padding=0)

        self.output_conv_T0 = nn.Conv2d(c, c, kernel_size=1, padding=0) # output T
        self.output_conv_T1 = nn.Conv2d(c, 1, kernel_size=1, padding=0) # output T
        self.tanh = nn.Tanh()

        # Define MLP layers V
        mlp_layers = []
        in_size = 2 * X * X  # Flattened input size based on c channels
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.ReLU(inplace=True))
            in_size = hidden_size

        self.hiddensize = mlp_hidden_sizes

        # Output layer to give a value between 0 and 1
        mlp_layers.append(nn.Linear(in_size, 1))
        mlp_layers.append(nn.Sigmoid())  # To output a value between 0 and 1
        
        self.mlp_v = nn.Sequential(*mlp_layers)

    
    def forward(self, x):

        board_channel = x[:,0]

        one_hot = torch.zeros((x.shape[0], 4, self.X, self.X), device=x.device, dtype=x.dtype)
        one_hot[:, 0, :, :] = (board_channel == -2)  # Channel for -2
        one_hot[:, 1, :, :] = (board_channel == -1)  # Channel for -1
        one_hot[:, 2, :, :] = (board_channel == 1)   # Channel for 1
        one_hot[:, 3, :, :] = (board_channel == 2)   # Channel for 2

        # Extract the active player channel (second channel)
        active_player_channel = x[:, 1, :, :].unsqueeze(1)  # Shape (B, 1, Y, X)

        # Concatenate the one-hot channels with the active player channel
        x = torch.cat([one_hot, active_player_channel], dim=1)  # Shape (B, 4, Y, X)

        #print(x[0])
    
        # Initial conv layer
        x = self.input_conv(x)
        x = self.relu(x)
        
        # Pass through N residual blocks
        x = self.resblocks(x)

        v = self.output_conv_v(x)
        v = self.relu(v)
        v = self.flatten(v)
        v = self.mlp_v(v)

        T = self.output_conv_T0(x)
        T = self.relu(T)
        T = self.output_conv_T1(T)
        T = self.tanh(T)
        
        return v,T
    
    def get_model_description(self):
        # Returns a dictionary description of the model
        return {
            'name': 'Q_V1_0',
            'n_resblock': self.N,
            'size': self.X,
            'n_feature': self.m,
            'n_channels': self.c,
            'hidden_sizes': '-'.join([str(hs) for hs in self.hiddensize])
        }

if __name__ == '__main__':
    # Example usage:
    m, X, N, c = 3, 32, 5, 64  # m input channels, X*X input size, N residual blocks, c channels
    mlp_hidden_sizes = [128, 64]  # Sizes of hidden layers in the MLP
    model = Q_V0_0(m, X, N, c, mlp_hidden_sizes)

    # Test forward pass with a random input tensor
    input_tensor = torch.randn((100, m, X, X))  # Batch size of 1
    output = model(input_tensor)
    print(output)
