from base_utils import *
from model_utils import *
from selfplay_test import *
import sys
import os
import datetime

wd = os.path.dirname(__file__)
boardsize = 8

m, X, B, c = 4, boardsize, 6, 96  # m input channels, X*X input size, N residual blocks, c channels
mlp_hidden_sizes = [256]  # Sizes of hidden layers in the MLP
Qmodel = Q_V0_1(m, X, B, c, mlp_hidden_sizes)

Qmodel.load_state_dict(torch.load('/home/mingshiyang/AI-Amazon-DQN/checkpoint.pth',weights_only=True))
Qmodel.eval()

print(Qmodel.get_model_description())
pi = Qmodel.get_model_description()
pi['time_created'] = datetime.datetime.now(datetime.UTC).strftime('UTC %Y-%m-%dT%H:%M:%S')
print(pi)
torch.save({'state_dict': Qmodel.state_dict(), 'parameter_info': pi}, os.path.join(wd,'data','Amazon_model.pth'))

pass