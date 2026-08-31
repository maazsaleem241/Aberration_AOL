# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 04:45:17 2023

@author: Super-depth imaging
"""

import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import mat73
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.functional import F

#%%
op_train_input = 0
nf = 64
n_basis = 1600
sz = 40

device = 'cuda'

#%% DealingComplex
def toComplex3(E): # input : 3D real/imag (2528,40,40) 
    return E[::2,:,:] + 1j*E[1::2,:,:] # output : 3D complex (1264,40,40)

def toComplex2(E): # input : 2D real/imag (2528,1600)
    return E[::2,:] + 1j*E[1::2,:] # output : 2D complex (1264,1600)

def toReal3(E): # input : 3D complex (1264,40,40)
    real_E = torch.zeros((n_basis*2,sz,sz),dtype = torch.float32).to(device)
    real_E[::2,:,:] = torch.real(E)
    real_E[1::2,:,:] = torch.imag(E)
    return real_E # output : 3D real/imag (2528,40,40)

def toReal2(E): # input : 2D complex (1264,1600)
    real_E = torch.zeros((n_basis*2,sz*sz),dtype = torch.float32).to(device)
    real_E[::2,:] = torch.real(E)
    real_E[1::2,:] = torch.imag(E)
    return real_E # output : 2D real/imag (2528,1600)

def toComplex_ab(E): # input : real/imag ab map (2,40,40)
    return E[0] + 1.0j*E[1] # output : complex ab map (40,40)

def toComplex_ab_batch(E): # input : (batch_size, 2, sz, sz)
    return E[:,0] + 1j*E[:,1] # output : complex batch (batch_size, sz, sz)

def toComplex_batch(E): # input : real/imag batch (batch_size, 2*n_basis, sz, sz)
    return E[:,::2] + 1j*E[:,1::2] # output : complex batch (batch_size, n_basis, sz, sz)
    

#%% 2D <-> 3D (only complex)
def to2(E): # input : 3D (1264,40,40) 
    return E.reshape(n_basis,sz*sz) # output : 2D (1264,1600)
def to2_batch(E):
    return E.reshape(E.shape[0], n_basis, sz*sz)
def to3(E): # input : 2D (1264,1600)
    return E.reshape(n_basis,sz,sz) # output : 3D (1264,40,40)

#%% useful transforms
def C2toR3(E):
    E = to3(E) 
    E = toReal3(E)
    return E

def R3toC2(E):
    E = toComplex3(E) 
    E = to2(E)
    return E

#%% Loss function
def PLCC(a, b):
    def variance(a, b):
        a = toComplex_batch(a)
        b = toComplex_batch(b)
        a_mean = torch.mean(a)
        b_mean = torch.mean(b)
        return abs(torch.mean((a-a_mean)*torch.conj(b-b_mean)))
    
    return -variance(a,b)/torch.sqrt(variance(a,a)*variance(b,b))
# Edited by Maaz: Modified for the new dataset
def plcc_loss(pred, target):
    """
    Standardized PLCC loss function tailored for 2-channel phase/aberration maps (Batch, 2, 40, 40).
    Returns the negative correlation coefficient for minimization.
    """
    # Use the targeted 2-channel aberration converter instead of the 3200-channel matrix one
    pred_c = toComplex_ab_batch(pred) if not pred.is_complex() else pred
    target_c = toComplex_ab_batch(target) if not target.is_complex() else target
    
    pred_mean = torch.mean(pred_c)
    target_mean = torch.mean(target_c)
    
    # Covariance calculation
    cov = torch.mean((pred_c - pred_mean) * torch.conj(target_c - target_mean))
    
    # Variance calculations
    var_pred = torch.mean(torch.abs(pred_c - pred_mean) ** 2)
    var_target = torch.mean(torch.abs(target_c - target_mean) ** 2)
    
    # Normalize with stability epsilon
    corr = torch.abs(cov) / (torch.sqrt(var_pred * var_target) + 1e-8)
    
    return -corr

def SimLoss(a, b):
    a = toComplex_batch(a)
    b = toComplex_batch(b)
    return -Similarity(a, b)

def RMS(ab_map):
    return torch.sqrt(torch.sum(ab_map**2))

def divLoss(truth, pred):
    pred_angle = torch.angle(toComplex_ab_batch(pred))
    truth_angle = torch.angle(toComplex_ab_batch(truth))
    return RMS(truth_angle - pred_angle)

#%% CASS
# Parameters
NAsz=20
matsz = 4*NAsz
[xx, yy] = torch.meshgrid(torch.arange(1,matsz+1),torch.arange(1,matsz+1))
[xx, yy] = torch.meshgrid(torch.arange(1,matsz+1),torch.arange(1,matsz+1))
xx = xx.type(torch.float32)
yy = yy.type(torch.float32)
cmask = torch.sqrt((xx - torch.mean(xx))**2 + (yy - torch.mean(yy))**2)/NAsz
cmask = cmask<=1
[kytmp, kxtmp] = torch.meshgrid(torch.arange(-NAsz,NAsz),torch.arange(-NAsz,NAsz))
n1 = (2*NAsz)**2 #1600
kytmp = kytmp.reshape(n1)
kxtmp = kxtmp.reshape(n1)
indx_dk = torch.zeros((n1, n1*4)); # 1600x6400
kfield = torch.ones((NAsz*2,NAsz*2)); # 40x40
kfield2 = F.pad(kfield,(NAsz,NAsz,NAsz,NAsz),'constant',0) # 80x80
for i in range(n1):
    indx_dk[i]=torch.roll(kfield2, shifts=(-kxtmp[i], -kytmp[i]), dims=(0,1)).permute(1,0).reshape(6400) #reshape->transposed?
    
def CASS(kkcmat): # (1600,1600) (kin,kout)
    dkk=torch.zeros(n1, n1*4, dtype=torch.complex64).to(device) # 1600x6400
    dkk[indx_dk>0] = kkcmat.to(device).reshape(1600*1600)
    return iFT2(torch.sum(dkk,dim=0).reshape(80,80).permute(1,0))

def CASS_I_batch(kkcmat): # (1600,1600) (kin,kout)
    dkk=torch.zeros(kkcmat.shape[0], n1, n1*4, dtype=torch.complex64) # 1600x6400
    dkk[:,indx_dk>0] = kkcmat.reshape(kkcmat.shape[0], 1600*1600).to('cpu')
    kimg = torch.sum(dkk,dim=1).reshape(dkk.shape[0],2*sz,2*sz).permute(0,2,1)
    return torch.mean(torch.abs(kimg.to(device))**2)*6400

def CASS_I(kkcmat):
    dkk=torch.zeros(n1, n1*4, dtype=torch.complex64).to(device) # 1600x6400
    dkk[indx_dk>0] = kkcmat.reshape(1600*1600)
    kimg = torch.sum(dkk,dim=0).reshape(2*sz,2*sz).permute(1,0)
    return torch.sum(torch.abs(kimg)**2).cpu()

#%% Required matrices
matdir = 'savedmat/'
if n_basis == 1264:
    
    p_mat = torch.tensor(mat73.loadmat(matdir+'Pmat.mat')['data_mat2'], dtype=torch.float32).permute(2,1,0).to(device)
    p_mat = to2(toComplex3(p_mat)) # (1264,1600)
    p_mat_T = p_mat.permute(1,0) # (1600,1264) 
    
    nzind = torch.tensor(mat73.loadmat(matdir+'nzind.mat')['nzind'], dtype=torch.int32) 
    
mask = torch.tensor(np.load(matdir+'mask.npy'))[20:60,20:60].to(device)
    
#%% Prediction
def get_pred(Input,m): # input : 3D real/imag (2526,40,40) 
    with torch.no_grad():
        m.eval()
        pred = toComplex_ab(m(Input.unsqueeze(dim=0)).squeeze())
        pred = torch.angle(pred)*mask
    return pred # output : ab PHASE map (40,40)

def get_pred_batch(Input,m):
    with torch.no_grad():
        m.eval()
        pred = toComplex_ab_batch(m(Input))
        pred = torch.angle(pred)*mask
    return pred # output : ab PHASE map (40,40)

def get_corr(Input, Output, m, op='out'):
    if op == 'out':
        truth = toComplex_ab(Output[0:2])
    elif op == 'in':
        truth = toComplex_ab(Output[2:4])
    pred_angle = get_pred(Input, m)
    pred = torch.exp(1j*pred_angle)

    return Similarity(truth, pred)

def get_corr_batch(Input, Output, m):
    truth = toComplex_ab_batch(Output)
    pred_angle = get_pred_batch(Input, m)
    pred = torch.exp(1j*pred_angle)

    return Similarity(truth, pred)

def Similarity(a,b):
    return abs(torch.sum(a*torch.conj(b))/torch.sqrt(torch.sum(a*torch.conj(a))*torch.sum(b*torch.conj(b)))).item()
    
#%% Correction
def correction(E,pred): # input : 3D real/imag (n_basis*2,sz,sz)
    E = toComplex3(E) # (n_basis,sz,sz)
    E = E/pred
    E = to2(E)
    return E # output : 2D complex (n_basis,sz*sz)

def correction_RR(E,pred):
    E = toComplex2(E)
    E = E/pred
    E = transpose_C3(E)
    E = E/torch.conj(pred)
    E = transpose_C3(E)
    E = to2(E)
    return E

def correction_batch(E,pred): # input : 4D real/imag (batch_size,n_basis*2,sz,sz), 4D real/image (batch_size,2,sz,sz)
    E = toComplex_batch(E).permute(1,0,2,3) # (batch_size,n_basis,sz,sz)
    pred = toComplex_ab_batch(pred)
    pred_angle = torch.angle(pred)
    pred = torch.exp(1j*pred_angle)
    E = E/pred 
    E = to2_batch(E.permute(1,0,2,3)) 
    return E # output : 3D complex (batch_size, n_basis,sz*sz)

def transpose_R3(E): # input : 3D real/imag (n_basis*2,sz,sz)
    E = toComplex3(E)
    E = to2(E)
    E = transpose_C2(E)
    E = C2toR3(E)
    return E # output : 3D real/imag (n_basis*2,sz,sz)

def transpose_C3(E): # input : complex 2D (n_basis,sz*sz)
    E = E.permute(1,2,0).reshape(n_basis,sz,sz)
    return E # output : complex 2D (n_basis,sz*sz)

def transpose_C2(E): # input : complex 2D (n_basis,sz*sz)
    if n_basis == 1600:
        E = E.permute(1,0)
    else:
        E = torch.matmul(p_mat_T, E)
        E = E.permute(1,0)
        E = E[nzind-1,:]
    return E # output : complex 2D (n_basis,sz*sz)

#%% Plot
def plot_pred(Input, Output, m, op='out'): # input : real/imag test_input (2*n_basis,sz,sz), test_output (2*40,40)
    if op == 'out':
        truth = toComplex_ab(Output[0:2])
    elif op == 'in':
        truth = toComplex_ab(Output[2:4])

    pred_angle = get_pred(Input, m)
    pred = torch.exp(1j*pred_angle)
    
    correlation = Similarity(pred, truth)
    
    plt.subplot(1,2,1)
    plt.imshow(torch.angle(truth).cpu().detach().numpy(), cmap='jet', vmin=-3.14, vmax=3.14)
    plt.title('ground truth')
    plt.gca().set_axis_off()
    
    plt.subplot(1,2,2)
    plt.imshow((pred_angle).cpu().detach().numpy(), cmap='jet', vmin=-3.14, vmax=3.14)
    if op == 'out':
        plt.title(f'predicted outab (corr : {correlation:.3f})')
    elif op == 'in':
        plt.title(f'predicted inab (corr : {correlation:.3f})')
    plt.gca().set_axis_off()
    plt.show()

#%% FT
def FT(x):
        return torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(x))) 
def iFT(x):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x))) 

def FT2(x):
    return torch.fft.fftshift(torch.fft.fft2(x))
        
def iFT2(x):
    return torch.fft.ifft2(torch.fft.ifftshift(x))

def tokspace(img): #(1600x40x40) (xin,kout)
    images = torch.zeros(n_basis,sz,sz,dtype=torch.complex64)
    for i in range(n_basis):
        images[i]=FT(img[i])
    return images #(kout,kin)

def toxspace(img):
    images = torch.zeros(n_basis,sz,sz,dtype=torch.complex64)
    for i in range(n_basis):
        images[i]=iFT(img[i])
    return images

def tokspace2(img): #(1600x40x40) (xin,kout)
    images = torch.zeros(n_basis,sz,sz,dtype=torch.complex64)
    for i in range(n_basis):
        images[i]=FT2(img[i])
    return images #(kout,kin)

def toxspace2(img):
    images = torch.zeros(n_basis,sz,sz,dtype=torch.complex64)
    for i in range(n_basis):
        images[i]=iFT2(img[i])
    return images

#%% ??
def toReal(complexTensor):
    return torch.stack((torch.real(complexTensor),torch.imag(complexTensor)),dim=0)

def saveMat(mat): #(3200,40,40) (xin,kout)
    E = toComplex2(mat) #(1600,40,40) (xin,kout)
    E = E.reshape(sz,sz,n_basis).permute(2,0,1) #(1600,40,40) (kout,xin)
    E = tokspace(E) #(1600,40,40) (kout,kin)
    E = E.reshape(n_basis,n_basis).permute(1,0)
    return E #(1600,1600) (kin,kout)

def toRiVec(mat): #(kin,kout) (1264,1600) (1600x1600)
    vec = toReal2(mat) # (2528,1600) (3200,1600)
    vec = vec.reshape(n_basis*2,sz,sz) #(2528x40x40)
    return vec 

def vecTrans(vec):
    return vec.permute(1,2,0).reshape(n_basis,40,40)

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
            CBDR2d(n_basis, nf, 3, 1, 1),
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

        self.fc = nn.Conv2d(nf, 2+2*op_train_input , kernel_size=1, stride=1) #40x40x2


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
    
#%%
def nor_correlate(img_1,img_2):
    nor_correlation = np.sum(img_2*np.conjugate(img_1))/(np.sqrt(np.sum(np.abs(img_1)**2))*np.sqrt(np.sum(np.abs(img_2)**2)))
    return nor_correlation
def phaseSync_(ref_img,measured_img,unit_input_img):
    
    tmp_img = measured_img/unit_input_img
    nor_correlation = nor_correlate(ref_img,tmp_img) # external function
    phase_diff = np.angle(nor_correlation)
    phaseSynced_img =  measured_img*np.exp(-1.0j*phase_diff)
    new_ref_img = ref_img + tmp_img * np.exp(-1.0j*phase_diff)
    return phaseSynced_img, new_ref_img, phase_diff 

def phaseSync(measured_data,input_data,itn):
   
    n_basis = len(measured_data)
    
    phaseSynced_data = measured_data*1 # to allocated two different variable on distinct memmory locations
    # ii_0 = n_basis//2
    ii_0 = 1000
    
    ref_img = phaseSynced_data[ii_0,:,:]/input_data[ii_0,:,:]
    phase_diff_data = np.zeros((n_basis+1,itn*2),dtype = np.float64)
    phase_diff_data[0,0] = ii_0
    a = np.random.permutation(n_basis)
    for i in range(itn):
        k = 1
        for ii in a:
            # phaseSynced_img, new_ref_img, phase_diff = phaseSync_(ref_img,phaseSynced_data[ii,:,:],input_data[ii,:,:])
            # phaseSynced_data[ii,:,:] = phaseSynced_img            
            # phase_diff_data[k,[2*i,(2*i+1)]] = [ii,phase_diff]
            # ref_img = new_ref_img
            
            tmp_img = phaseSynced_data[ii,:,:]/input_data[ii,:,:]
            nor_correlation = nor_correlate(ref_img,tmp_img)
            phase_diff = np.angle(nor_correlation)
            phaseSynced_img = phaseSynced_data[ii,:,:]*np.exp(-1.0j*phase_diff)
            ref_img = ref_img + tmp_img*np.exp(-1.0j*phase_diff)
            
            phaseSynced_data[ii,:,:] = phaseSynced_img            
            phase_diff_data[k,[2*i,(2*i+1)]] = [ii,phase_diff]
            k = k + 1
            
    return phaseSynced_data, phase_diff_data
