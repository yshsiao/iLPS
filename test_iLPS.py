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
thre_conf = 0.999
os = 16
D = md.iLPSclassifier(input_size = w)
D.load_state_dict(torch.load('model/os1_classifier_w41.w'))
L1error = 0
L2error = 0
L1errorm = 0
L2errorm = 0
num_testdata =  dataset.__len__()
d_shift = 13 #need to be modified
for i, (train_data) in enumerate(train_loader):
    #print train_data['X'][0], train_data['Y'][0],train_data['offset'][0]
    #uinput = Variable(train_data['X']).float()
    if i % 50 == 0:
        print i
    totalinput = Variable(train_data['testos1']).float()
    target = Variable(train_data['testdist']).float()
    length = totalinput.shape[2]
    #print totalinput.shape
    possible_ontime = np.zeros((1,length))
    possible_early = np.zeros((1,length))
    possible_late = np.zeros((1,length))
    possible_noise = np.zeros((1,length))
    possible_pos = np.zeros((1,length))
    posidx = np.arange(1,length+1,1)
    for j in range(length-w):
        output = D(totalinput[:,:,j:j+w])
        #print output,j
        possible_ontime[0,j+int((w-1)/2)] = output[0,0,2].detach().numpy()
        possible_early[0,j+int((w-1)/2)] = output[0,0,1].detach().numpy()
        possible_late[0,j+int((w-1)/2)] = output[0,0,3].detach().numpy()
        possible_noise[0,j+int((w-1)/2)] = output[0,0,0].detach().numpy()
    for k in range(int((w-1)/2),length-int((w-1)/2)):
        possible_pos[0,k] = possible_ontime[0,k]*possible_early[0,k-int((w-1)/2)]*possible_late[0,k+int((w-1)/2)]*(1-possible_noise[0,k])
    maxidx = np.argmax(possible_pos)
    confident_idx = np.zeros((1,length))
    for k in range(int((w-1)/2),length-int((w-1)/2)):
        if possible_pos[0,k] > possible_pos[0,maxidx]*thre_conf:
            confident_idx[0,k] = 1
    result_pos = sum(sum(possible_pos*posidx*confident_idx))/sum(sum(possible_pos*confident_idx))*3.75/os
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

