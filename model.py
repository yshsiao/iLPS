import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import math
import cmath
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import random
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class iLPSclassifier(nn.Module):
    def __init__(self, input_size = 81, hidden1 = 32, hidden2=16, log=1, classnum = 4):
        super(iLPSclassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2,classnum)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.log = log
    
    def forward(self, xx):
        if self.log == 1:
            out = 10*torch.log(torch.abs(xx))
        out = self.fc1(xx)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

