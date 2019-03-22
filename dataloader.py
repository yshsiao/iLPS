import torch
import torch.utils.data as data
import glob
import librosa
import numpy as np


class iLPS_trainloader(data.Dataset):

    def __init__(self, dataPath):

        self.data = np.load(dataPath)
        self.data_len = len(self.data['O'][0,:])

    def __getitem__(self, index):

        N1 = np.zeros((1,self.data_len))
        N2 = np.zeros((1,self.data_len))
        E = np.zeros((1,self.data_len))
        L = np.zeros((1,self.data_len))
        O = np.zeros((1,self.data_len))
        N1[0,:] = self.data['N1'][index,:]
        N2[0,:] = self.data['N2'][index,:]
        E[0,:] = self.data['E'][index,:]
        L[0,:] = self.data['L'][index,:]
        O[0,:] = self.data['O'][index,:]
        pair = {'N1':N1, 'N2':N2, 'E':E, 'L':L, 'O':O}
        return pair

    def __len__(self):
        return len(self.data['O'][:,0])

class iLPS_testloader(data.Dataset):

    def __init__(self, dataPath):

        self.data = np.load(dataPath)
        self.data_len = len(self.data['testos1'][0,:])


    def __getitem__(self, index):

        testos1 = np.zeros((1,self.data_len))
        testos2 = np.zeros((1,self.data_len))
        testos4 = np.zeros((1,self.data_len))
        testos8 = np.zeros((1,self.data_len))
        testos1[0,:] = self.data['testos1'][index,:]
        testos2[0,:] = self.data['testos2'][index,:]
        testos4[0,:] = self.data['testos4'][index,:]
        testos8[0,:] = self.data['testos8'][index,:]
        offset = self.data['testdist'][index,:]
        pair = {'testos1':testos1,'testos2':testos2,'testos4':testos4,'testos8':testos8,'testdist':offset}
        return pair

    def __len__(self):
        return len(self.data['testos1'][:,0])



