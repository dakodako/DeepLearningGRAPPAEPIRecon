#%%
import scipy.io as sio
import scipy as sp 
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import linalg as LA 
# pytorch
# import torch 
# import torch.nn as nn 
# import torch.nn.functional as F 
# from torchsummary import summary
# from torchviz import make_dot
# import torch.optim as optim
import pandas as pd
import warnings
import nvgpu
from PIL import Image
import tensorflow as tf 
import numpy.matlib
from tensorflow.examples.tutorials.mnist import input_data
import time 
import os

# %%
def mosaic(img, num_row, num_col, fig_num, clim, fig_title='', num_rot=0, fig_size=(18, 16)):
    
    
    fig = plt.figure(fig_num, figsize=fig_size)
    fig.patch.set_facecolor('black')

    img = np.abs(img)
    img = img.astype(float)
        
    if img.ndim < 3:
        img = np.rot90(img, k=num_rot, axes=(0,1))
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
        title_str = fig_title
        plt.savefig(title_str + '.png')

    else: 
        img = np.rot90(img, k=num_rot,axes=(1,2))
        
        if img.shape[0] != (num_row * num_col):
            print('sizes do not match')    
        else:   
            img_res = np.zeros((img.shape[1]*num_row, img.shape[2]*num_col))
            
            idx = 0
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[1] : (r+1)*img.shape[1], c*img.shape[2] : (c+1)*img.shape[2]] = img[idx,:,:]
                    idx = idx + 1
               
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
        plt.title(fig_title, color='white')
        title_str = fig_title
        plt.savefig(title_str + '.png')
#%%
# Notes
#-------
# 1) Only use odd kernel size
# 2) samples.shape and acs.shape should be [num_chan, N1, N2]
# 3) Both samples and acs should be zero filled, and have the same matrix size as the original fully sampled image
# 4) The parity of each image dimension must match the parity of the corresponding acs dimension
# 5) This function doesn't substitute the ACS back after reconstruction

def grappa(samples, acs, Rx, Ry, num_acs, shift_x, shift_y, kernel_size=np.array([3,3]), lambda_tik=0):
    
    # Set Initial Parameters
    #------------------------------------------------------------------------------------------
    [num_chan, N1, N2] = samples.shape
    N = np.array([N1, N2]).astype(int)
    
    acs_start_index_x = N1//2 - num_acs[0]//2 #inclusive
    acs_start_index_y = N2//2 - num_acs[1]//2 #inclusive
    acs_end_index_x = np.int(np.ceil(N1/2)) + num_acs[0]//2
    acs_end_index_y = np.int(np.ceil(N2/2)) + num_acs[1]//2
    
    kspace_sampled = np.zeros(samples.shape, dtype=samples.dtype)
    kspace_sampled[:] = samples[:]
    
    kspace_acs = np.zeros(acs.shape, dtype=acs.dtype)
    kspace_acs[:] = acs[:]
    
    kspace_acs_crop = np.zeros([num_chan, num_acs[0], num_acs[1]], dtype=acs.dtype)
    kspace_acs_crop[:,:,:] = kspace_acs[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]
    
    #Kernel Side Size
    kernel_hsize = (kernel_size // 2).astype(int)

    #Padding
    pad_size = (kernel_hsize * [Rx,Ry]).astype(int)
    N_pad = N + 2 * pad_size

    # Beginning/End indices for kernels in the acs region
    ky_begin_index = (Ry*kernel_hsize[1]).astype(int)
    ky_end_index = (num_acs[1] - Ry*kernel_hsize[1] - 1 - np.amax(shift_y)).astype(int)

    kx_begin_index = (Rx*kernel_hsize[0]).astype(int)
    kx_end_index = (num_acs[0] - Rx*kernel_hsize[0] - 1 - np.amax(shift_x)).astype(int)

    # Beginning/End indices for kernels in the full kspace
    Ky_begin_index = (Ry*kernel_hsize[1]).astype(int)
    Ky_end_index = (N_pad[1] - Ry*kernel_hsize[1] - 1).astype(int)

    Kx_begin_index = (Rx*kernel_hsize[0]).astype(int)
    Kx_end_index = (N_pad[0] - Rx*kernel_hsize[0] - 1).astype(int)

    # Count the number of kernels that fit the acs region
    ind = 0
    for i in range(ky_begin_index, ky_end_index+1):
        for j in range(kx_begin_index, kx_end_index+1):
            ind +=1

    num_kernels = ind

    # Initialize right hand size and acs_kernel matrices
    target_data = np.zeros([num_kernels, num_chan, Rx, Ry], dtype=samples.dtype)
    kernel_data = np.zeros([num_chan, kernel_size[0], kernel_size[1]], dtype=samples.dtype)
    acs_data = np.zeros([num_kernels, kernel_size[0] * kernel_size[1] * num_chan], dtype=samples.dtype)

    # Get kernel and target data from the acs region
    #------------------------------------------------------------------------------------------
    print('Collecting kernel and target data from the acs region')
    print('------------------------------------------------------------------------------------------')
    kernel_num = 0
    
    for ky in range(ky_begin_index, ky_end_index + 1):
        print('ky: ' + str(ky))
        for kx in range(kx_begin_index, kx_end_index + 1):
            # Get kernel data
            for nchan in range(0,num_chan):
                kernel_data[nchan, :, :] = kspace_acs_crop[nchan, 
                                                           shift_x[nchan] + kx - (kernel_hsize[0]*Rx): shift_x[nchan] + kx + (kernel_hsize[0]*Rx)+1: Rx, 
                                                           shift_y[nchan] + ky - (kernel_hsize[1]*Ry): shift_y[nchan] + ky + (kernel_hsize[1]*Ry)+1: Ry]

            acs_data[kernel_num, :] = kernel_data.flatten()

            # Get target data
            for rx in range(0,Rx):
                for ry in range(0,Ry):
                    if rx != 0 or ry != 0:
                        for nchan in range(0,num_chan):
                            target_data[kernel_num,:,rx,ry] = kspace_acs_crop[:,
                                                                              shift_x[nchan] + kx - rx,
                                                                              shift_y[nchan] + ky - ry]

            # Move to the next kernel
            kernel_num += 1
            
    print()

    # Tikhonov regularization
    #------------------------------------------------------------------------------------------
    U, S, Vh = sp.linalg.svd(acs_data, full_matrices=False)
    
    print('Condition number: ' + str(np.max(np.abs(S))/np.min(np.abs(S))))
    print()
    
    S_inv = np.conjugate(S) / (np.square(np.abs(S)) + lambda_tik)
    acs_data_inv = np.transpose(np.conjugate(Vh)) @ np.diag(S_inv) @ np.transpose(np.conjugate(U));

    # Get kernel weights
    #------------------------------------------------------------------------------------------
    print('Getting kernel weights')
    print('------------------------------------------------------------------------------------------')
    kernel_weights = np.zeros([num_chan, kernel_size[0] * kernel_size[1] * num_chan, Rx, Ry], dtype=samples.dtype)

    for rx in range(0,Rx):
        print('rx: ' + str(rx))
        for ry in range(0,Ry):
            print('ry: ' + str(ry))
            if rx != 0 or ry != 0:
                for nchan in range(0,num_chan):
                    print('Channel: ' + str(nchan+1))
                    if lambda_tik == 0:
                        kernel_weights[nchan, :, rx, ry], resid, rank, s = np.linalg.lstsq(acs_data,target_data[:, nchan, rx, ry], rcond=None)
                    else:
                        kernel_weights[nchan, :, rx, ry] = acs_data_inv @ target_data[:, nchan, rx, ry]
                        
    print()

    # Reconstruct unsampled points
    #------------------------------------------------------------------------------------------
    print('Reconstructing unsampled points')
    print('------------------------------------------------------------------------------------------')
    kspace_recon = np.pad(kspace_sampled, ((0, 0), (pad_size[0],pad_size[0]), (pad_size[1],pad_size[1])), 'constant')
    data = np.zeros([num_chan, kernel_size[0] * kernel_size[1]], dtype=samples.dtype)

    for ky in range(Ky_begin_index, Ky_end_index+1, Ry):
        print('ky: ' + str(ky))
        for kx in range(Kx_begin_index, Kx_end_index+1, Rx):

            for nchan in range(0,num_chan):
                data[nchan, :] = (kspace_recon[nchan,
                                               shift_x[nchan] + kx - (kernel_hsize[0]*Rx): shift_x[nchan] + kx + (kernel_hsize[0]*Rx)+1: Rx, 
                                               shift_y[nchan] + ky - (kernel_hsize[1]*Ry): shift_y[nchan] + ky + (kernel_hsize[1]*Ry)+1: Ry]).flatten()


            for rx in range(0,Rx):
                for ry in range(0,Ry):
                    if rx != 0 or ry != 0:
                        for nchan in range(0,num_chan):
                            interpolation = np.dot(kernel_weights[nchan, :, rx, ry] , data.flatten())
                            kspace_recon[nchan, shift_x[nchan] + kx - rx, shift_y[nchan] + ky - ry] = interpolation

    # Get the image back
    #------------------------------------------------------------------------------------------
    kspace_recon = kspace_recon[:, pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]  
    img_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))
    
    print()
    print('GRAPPA reconstruction complete.')
    
    return kspace_recon, img_grappa


# %%
N1 = 86
N2 = 98
num_acs = (32,N2)
num_chan = 32
Rx = 2
acs_start_index_x = N2//2 - num_acs[0]//2 #inclusive
acs_start_index_y = N2//2 - num_acs[1]//2 #inclusive
acs_end_index_x = np.int(np.ceil(N2/2)) + num_acs[0]//2
acs_end_index_y = np.int(np.ceil(N2/2)) + num_acs[1]//2
    
#%%
kspace_data = sio.loadmat('../MRI_data/MRI_data/raw_data/new_image_data_meas1_slc5_noSeg.mat')
kspace_undersampled_data = kspace_data['new_image_data_meas1_slc5_noSeg']
kspace_refscan = sio.loadmat('../MRI_data/MRI_data/raw_data/new_refscan_data_slc5_noSeg.mat')
kspace_refscan_data_cropped = kspace_refscan['new_refscan_data_slc5_noSeg']
#%%
kspace_undersampled_data = np.transpose(kspace_undersampled_data,[1,2,0])
kspace_refscan_data_cropped = np.transpose(kspace_refscan_data_cropped, [1,2,0])
#%%
kspace_refscan_data_zf = np.zeros(kspace_undersampled_data.shape)
kspace_refscan_data_zf[:,acs_start_index_x:acs_end_index_x,:] = kspace_refscan_data_cropped
# kspace_refscan = sio.loadmat('refscan_data_slice5_cropped.mat')
# kspace_refscan_data_cropped = kspace_refscan['refscan_data_slice5']
# %%
mosaic(kspace_refscan_data_cropped, 4,8,1,[0,0.001], fig_size =(15,15), num_rot=0)

#%%
mosaic(kspace_refscan_data_zf, 4, 8, 1, [0,0.001], fig_size=(15,15), num_rot=0)

# %%
mosaic(kspace_undersampled_data, 4,8,1,[0, 0.0005], fig_size=(15,15), num_rot = 0)

# %%
grappa_shift_x = np.zeros(num_chan, dtype=int)
grappa_shift_y = np.zeros(num_chan, dtype=int)

# %%
grappa_coil_images_without_sub = np.zeros(kspace_undersampled_data.shape, dtype=kspace_undersampled_data.dtype)
grappa_coil_kspaces_without_sub = np.zeros(kspace_undersampled_data.shape, dtype=kspace_undersampled_data.dtype)

grappa_coil_kspaces_without_sub, grappa_coil_images_without_sub = grappa(kspace_undersampled_data,kspace_refscan_data_zf, 
                                                                         Rx, 1, num_acs, grappa_shift_x, grappa_shift_y,kernel_size=np.array([3, 5]), lambda_tik=0)


# %%
acs_mask = np.zeros((num_chan, N1, N2), dtype=bool)
acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = True

grappa_coil_kspaces_sub = (grappa_coil_kspaces_without_sub * np.invert(acs_mask).astype(int)) + kspace_refscan_data_zf[:,:,:]

grappa_coil_images_sub = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grappa_coil_kspaces_sub, axes=(1,2)), axes=(1,2)), axes=(1,2))

# Visualize
print('Axis Labels: ' + '[num_coils, N1, N2]')
print('GRAPPA Coil Images Shape: ' + str(grappa_coil_images_sub.shape))
print('GRAPPA Coil Images Data Type: ' + str(grappa_coil_images_sub.dtype))
print()

grappa_rssq_image_sub = np.sqrt(np.sum(np.square(np.abs(grappa_coil_images_sub)),axis=0))

# Visualize
print('Axis Labels: ' + '[N1, N2]')
print('GRAPPA RSSQ Image Shape: ' + str(grappa_rssq_image_sub.shape))
print('GRAPPA RSSQ Image Data Type: ' + str(grappa_rssq_image_sub.dtype))
print()

#comparison_grappa = np.concatenate((np.expand_dims(image_original, axis=0), np.expand_dims(grappa_rssq_image_sub, axis=0)), axis = 0)
#rmse_grappa_sub = np.sqrt(np.sum(np.square(image_original - grappa_rssq_image_sub))) / np.sqrt(np.sum(np.square(image_original)))

#print('rmse_grappa: ' + str(rmse_grappa_sub))
#print()
#print('Original Image, Grappa Image')
#mosaic(comparison_grappa , 1, 2, 1, [0,1], fig_size=(15,15), num_rot=0)

# %%
acs_mask = np.zeros((num_chan, N1, N2), dtype=bool)
acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = True

kspace = kspace_undersampled_data * np.invert(acs_mask).astype(int) + kspace_refscan_data_zf[:,:,:]
kspace = np.rot90(kspace,k=1,axes=(1,2))
mosaic(kspace, 4, 8, 1, [0,0.001], fig_size=(15,15), num_rot=0)

# %%
