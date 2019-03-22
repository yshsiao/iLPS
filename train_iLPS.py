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
bsize = 100
dataset = dl.iLPS_trainloader('data/train20000w41_os1.npz')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)
#Model parameters
learning_rate = 1e-2
num_epochs = 10
w = 41

# open the log file
f = open("log/os1_trainw41.log","w")
f.write("Parameter: epoch=%s \n batch size: %s, learning rate D:%s \n train with os=32 only record os1,2,4,8" %
			  (num_epochs, bsize, learning_rate))
f.flush()

D = md.iLPSclassifier(input_size = w)
#D.load_state_dict(torch.load('model/os1_classifier_w41.w'))
L1criterion = nn.L1Loss()
L2criterion = nn.MSELoss()
CEcriterion = nn.CrossEntropyLoss()
d_optim = optim.Adam(D.parameters(), lr=learning_rate)
exp_lr_scheduler_D = optim.lr_scheduler.StepLR(d_optim, step_size=2, gamma=0.2)
den1 = 0
den2 = 0

for epoch in range(num_epochs):
    exp_lr_scheduler_D.step()
    for i, (train_data) in enumerate(train_loader):
        #print train_data['X'][0], train_data['Y'][0],train_data['offset'][0]
        # I define noise is 0, early is 1, ontime is 2, late is 3
        # N1, N2 are the noise(before and after), E is early, O is ontime, L is late part.
        z1 = torch.zeros([bsize])        
        dinput = Variable(train_data['N1']).float()
        target = Variable(z1).long()
        D.zero_grad()
        d_decision = D(dinput)
        d_decision = torch.squeeze(d_decision,1)
        d_error1 = CEcriterion(d_decision, target)
        d_error1.backward() 
        d_optim.step()
        den1 = extract(d_error1)[0]

        z1 = torch.zeros([bsize])
        dinput = Variable(train_data['N2']).float()
        target = Variable(z1).long()
        D.zero_grad()
        d_decision = D(dinput)
        d_decision = torch.squeeze(d_decision,1)
        d_error2 = CEcriterion(d_decision, target)
        d_error2.backward() 
        d_optim.step()
        den2 = extract(d_error2)[0]

        o1 = torch.ones([bsize])
        dinput = Variable(train_data['E']).float()
        target = Variable(o1).long()
        D.zero_grad()
        d_decision = D(dinput)
        d_decision = torch.squeeze(d_decision,1)
        d_error3 = CEcriterion(d_decision, target)
        d_error3.backward() 
        d_optim.step()
        dee = extract(d_error3)[0]

        o1 = torch.ones([bsize])
        dinput = Variable(train_data['O']).float()
        target = Variable(o1*2).long()
        D.zero_grad()
        d_decision = D(dinput)
        d_decision = torch.squeeze(d_decision,1)
        d_error4 = CEcriterion(d_decision, target)
        d_error4.backward() 
        d_optim.step()
        deo = extract(d_error4)[0]

        o1 = torch.ones([bsize])
        dinput = Variable(train_data['L']).float()
        target = Variable(o1*3).long()
        D.zero_grad()
        d_decision = D(dinput)
        d_decision = torch.squeeze(d_decision,1)
        d_error5 = CEcriterion(d_decision, target)
        d_error5.backward() 
        d_optim.step()
        dela = extract(d_error5)[0]

        print("Epoch %s: Iteration: %s \n D(n1err: %s,n2err: %s,Eerr: %s,Oerr: %s,Lerr: %s) " 
                % (epoch,i, den1, den2, dee, deo, dela))
        f.write("Epoch %s: Iteration: %s \n D(n1err: %s,n2err: %s,Eerr: %s,Oerr: %s,Lerr: %s) \n" 
                %(epoch,i, den1, den2, dee, deo, dela))
        f.flush()	

torch.save(D.state_dict(),'model/os1_classifier_w41.w')
#G = torch.load('G_model.w')
f.close()


