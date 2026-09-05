# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:24:54 2023

@author: Super-depth imaging
"""
import os
os.chdir('D:/eunyoung/DeepCLASS')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.nn import init
from tqdm import trange
from tqdm import tqdm
import time
import math
from scipy import io
from PIL import Image
import cv2
import mat73
import os
import itf as f
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<>"
#torch.cuda.empty_cache()

#%% Parameters
op_data_inab = 0 # 0:only outab, 1:both out,inab
op_train_inab = 0 # 0:train only outab, 1:train both out,inab
op_angle = 0 # output angle 

nf = 64
n_basis = 1600
sz = 40
n_output = 2-op_angle+2*op_train_inab

num_epoch = 100
batch_size = 400
lr = 0.0001
#R = 1e-5

loss_function = 'PLCC'

n_test = 100

#%% dir
datadir = 'train_dataset/kk_full_1000_sm_onlyout/'
savedir = datadir + 'save/' + loss_function #+ '_div_'+str(R)

#%% UNet
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def CBDR2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1):
            layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=output_channel),
                nn.Dropout(0.1),
                nn.ReLU()
            )
            return layer

		# Contracting path
        self.conv1 = nn.Sequential(
            CBDR2d((2-op_angle)*n_basis, nf, 3, 1, 1),
            CBDR2d(nf, nf, 3, 1, 1),
            CBDR2d(nf, nf, 3, 1, 1)
        )	
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #20x20x64

        self.conv2 = nn.Sequential(
            CBDR2d(nf, nf*2, 3, 1, 1),
            CBDR2d(nf*2, nf*2, 3, 1, 1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #10x10x128

        self.conv3 = nn.Sequential(
            CBDR2d(nf*2, nf*4, 3, 1, 1),
            CBDR2d(nf*4, nf*4, 3, 1, 1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #5x5x256

        #BottleNeck
        self.bottleNeck = nn.Sequential(
            CBDR2d(nf*4, nf*8, 3, 1, 1),
            CBDR2d(nf*8, nf*8, 3, 1, 1),
        ) # 5x5x512

		# Expanding path 
        self.upconv1 = nn.ConvTranspose2d(in_channels=nf*8, out_channels=nf*4, kernel_size=2, stride=2)

        self.ex_conv1 = nn.Sequential(
            CBDR2d(nf*8, nf*4, 3, 1, 1),
            CBDR2d(nf*4, nf*4, 3, 1, 1)
        ) # 10x10x256

        self.upconv2 = nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=2, stride=2)

        self.ex_conv2 = nn.Sequential(
            CBDR2d(nf*4, nf*2, 3, 1, 1),
            CBDR2d(nf*2, nf*2, 3, 1, 1)
        ) # 20x20x128

        self.upconv3 = nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf, kernel_size=2, stride=2)

        self.ex_conv3 = nn.Sequential(
            CBDR2d(nf*2, nf, 3, 1, 1),
            CBDR2d(nf, nf, 3, 1, 1)
        ) # 20x20x64

        self.fc = nn.Conv2d(nf, n_output , kernel_size=1, stride=1) #40x40x2


    def forward(self, x):
    	# Contracting path
        layer1 = self.conv1(x)
        out = self.pool1(layer1)

        layer2 = self.conv2(out)
        out = self.pool2(layer2)

        layer3 = self.conv3(out)
        out = self.pool3(layer3)

		# bottleneck 
        bottleNeck = self.bottleNeck(out)

		# Expanding path
        upconv1 = self.upconv1(bottleNeck)
        cat1 = torch.cat((transforms.CenterCrop((upconv1.shape[2], upconv1.shape[3]))(layer3), upconv1), dim=1)
        ex_layer1 = self.ex_conv1(cat1)

        upconv2 = self.upconv2(ex_layer1)
        cat2 = torch.cat((transforms.CenterCrop((upconv2.shape[2], upconv2.shape[3]))(layer2), upconv2), dim=1)
        ex_layer2 = self.ex_conv2(cat2)

        upconv3 = self.upconv3(ex_layer2)
        cat3 = torch.cat((transforms.CenterCrop((upconv3.shape[2], upconv3.shape[3]))(layer1), upconv3), dim=1)
        out = self.ex_conv3(cat3)

        out = self.fc(out)
        return out
    
#%% Check GPU
device = 'cuda'

#%% DataLoad
s = time.time()
train_input = torch.tensor(mat73.loadmat(datadir + 'train/data_mat.mat')['data_mat'], dtype=torch.float32).permute(0,3,2,1)
train_output = torch.tensor(mat73.loadmat(datadir + 'train/truth_mat.mat')['truth_mat'], dtype=torch.float32).permute(0,3,2,1)[:,0:2,:,:]

test_input = torch.tensor(mat73.loadmat(datadir + 'test/data_mat.mat')['data_mat'], dtype=torch.float32).permute(0,3,2,1)
test_output = torch.tensor(mat73.loadmat(datadir + 'test/truth_mat.mat')['truth_mat'], dtype=torch.float32).permute(0,3,2,1)[:,0:2,:,:]
e = time.time()
print(f'{(e-s)/60} min for data loading')

#%%
train_dataset = TensorDataset(train_input, train_output)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_input, test_output)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
e = time.time()

#%% Training
m = UNet()
m = nn.DataParallel(m)
m.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=lr)
s = time.time()

if loss_function == 'mse':
    loss_ = F.mse_loss
elif loss_function == 'PLCC':
    loss_ = f.PLCC
elif loss_function == 'RMS':
    loss_ = f.divLoss

loss_hist = []
test_loss_hist=[]

for epoch in trange(num_epoch):
    loss = 0
    count = 0
    m.train()
    for Input, Output in train_dataloader:
        count += 1
        Input = Input.to(device)
        Output = Output.to(device)
        optimizer.zero_grad()
        Output_preds = m(Input)
        loss_val = loss_(Output,Output_preds) #+R*f.divLoss(Output,Output_preds)#- R*(torch.log10(f.CASS_I_batch(f.correction_batch(Input,Output_preds)))-9)/2
        #Input = Input.to('cpu')
        #torch.cuda.empty_cache()
        loss_val.backward()
        optimizer.step()
        loss += loss_val.item()
        
    loss_hist.append(loss/count)

    with torch.no_grad():      
        test_loss = 0
        test_count = 0
        m.eval()
        for test_input, test_output in test_dataloader:
            test_count += 1
            test_input = test_input.to(device)
            test_output = test_output.to(device)
            test_pred = m(test_input)
            test_loss_val = loss_(test_output, test_pred) #+R*f.divLoss(test_output,test_pred)#- R*(torch.log10(f.CASS_I_batch(f.correction_batch(test_input,test_pred)))-9)/2
            test_loss += test_loss_val.item()
            
        test_loss_hist.append(test_loss/test_count)

e = time.time()
print(f'{(e-s)/60} min for training')
np.save(savedir+'loss_hist.npy',[loss_hist,test_loss_hist])

#%%
plt.plot(loss_hist, label ='train')
plt.plot(test_loss_hist, label = 'test')
plt.savefig(savedir+f'_{nf}.png')
plt.legend()
plt.show()

#%% Save Model
torch.save(m.module, savedir+f'_{nf}.pt')




















