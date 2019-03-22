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
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.io as sio
import model as md
import dataloader as dl
# support function
def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

# Load training data here: need to modify dataloader
bsize = 1
dataset = dl.iLPS_testloader('data/testw41_1511.npz')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)
w = 41
thre_prob = 0.5
D = md.iLPSclassifier(input_size = w)
D.load_state_dict(torch.load('model/os1_classifier_w41.w'))
L1error = 0
L2error = 0
L1errorm = 0
L2errorm = 0
num_testdata =  dataset.__len__()
for i, (train_data) in enumerate(train_loader):
    #print train_data['X'][0], train_data['Y'][0],train_data['offset'][0]
    #uinput = Variable(train_data['X']).float()
    if i % 50 == 0:
        print i
    totalinput = Variable(train_data['testos1']).float()
    target = Variable(train_data['testdist']).float()
    length = totalinput.shape[2]
    #print totalinput.shape
    possible_pos = np.zeros((1,length))
    posidx = np.arange(1,length+1,1)
    conti = 0
    visit = 0
    for j in range(length-w):
        output = D(totalinput[:,:,j:j+w])
        #print output,j
        if output[0,0,2]>output[0,0,3] and output[0,0,2]>output[0,0,0] and output[0,0,2]>output[0,0,1] and output[0,0,2]> thre_prob and (visit+conti)%2 == 0:
            possible_pos[0,j+40] = output[0,0,2].detach().numpy()
            conti = 1
            visit = 1
        else:
            conti = 0
    result_pos = sum(sum(possible_pos*posidx))/sum(sum(possible_pos))*3.75/16
    diff = result_pos - target.detach().numpy()
    #print diff
    maxidx = np.argmax(totalinput.detach().numpy())
    max_pos = maxidx*3.75/16
    diffmax = max_pos - target.detach().numpy()
    #print diffmax
    L1error = L1error + np.abs(diff)
    L2error = L2error + diff*diff
    L1errorm = L1errorm + np.abs(diffmax)
    L2errorm = L2errorm + diffmax*diffmax

print 'NN result:'
print L1error/num_testdata
print L2error/num_testdata
print 'Find max result:'
print L1errorm/num_testdata
print L2errorm/num_testdata

#G = torch.load('G_model.w')


