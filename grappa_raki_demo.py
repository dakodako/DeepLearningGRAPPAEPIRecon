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
#%%
import pandas as pd
import warnings
import nvgpu
from PIL import Image

# %%
import tensorflow as tf 
import numpy.matlib
from tensorflow.examples.tutorials.mnist import input_data
import time 
import os

# %%
# Notes
#-------
# 1) Given an input image of shape [num_images, N1, N2], concatenates all images in a mosaic and displays it
# 2) num_images must match num_row * num_col

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
dt2 = sio.loadmat('SPARK-master/img_grappa_32chan.mat')
img_coils = dt2['IMG']

# %%
img_coils = np.rot90(np.transpose(img_coils, (2,0,1)), k =-1, axes = (1,2))
img_coils = img_coils[:,:,:]
[num_chan, N1, N2] = img_coils.shape
mosaic(img_coils, 4,8,1,[0,1],fig_size = (15,15),num_rot = 0)


# %%
image_original = np.sqrt(np.sum(np.square(np.abs(img_coils)), axis = 0))
mosaic(image_original, 1,1,1,[0,1],fig_size = (10,10), num_rot = 0)

# %%
num_acs = (30,N2) # 30 lines of autocalibration line
Rx = 6
kspace_fully_sampled = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_coils, axes=(1,2)), axes=(1,2)), axes=(1,2))
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('Kspace Shape: ' + str(kspace_fully_sampled.shape))
print('Data Type: ' + str(kspace_fully_sampled.dtype))

mosaic(kspace_fully_sampled[:,:,:], 4, 8, 1, [0,50], fig_size=(15,15), num_rot=0)

# %%
kspace_original = np.sqrt(np.sum(np.square(np.abs(kspace_fully_sampled)),axis=0))

# Visualize
print('Axis Labels: ' + '[N1, N2]')
print('Kspace Shape: ' + str(kspace_original.shape))
print('Kspace Type: ' + str(kspace_original.dtype))

mosaic(kspace_original, 1, 1, 1, [0,150], fig_size=(10,10), num_rot=0)

# %%
acs_start_index_x = N1//2 - num_acs[0]//2 # inclusive
acs_start_index_y = N2//2 - num_acs[1]//2 # inclusive
acs_end_index_x = np.int(np.ceil(N1/2)) + num_acs[0]//2 # exclusive
acs_end_index_y = np.int(np.ceil(N2/2)) + num_acs[1]//2 # exclusive
    
kspace_acs_zero_filled = np.zeros(kspace_fully_sampled.shape, dtype=kspace_fully_sampled.dtype)
kspace_acs_cropped = np.zeros([num_chan, num_acs[0], num_acs[1]], dtype=kspace_fully_sampled.dtype)

kspace_acs_zero_filled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = kspace_fully_sampled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]
kspace_acs_cropped[:,:,:] = kspace_fully_sampled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]

# Visualize
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('kspace_acs_zero_filled Shape: ' + str(kspace_acs_zero_filled.shape))
print('kspace_acs_zero_filled Data Type: ' + str(kspace_acs_zero_filled.dtype))
print()

# Visualize
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('kspace_acs_cropped Shape: ' + str(kspace_acs_cropped.shape))
print('kspace_acs_cropped Data Type: ' + str(kspace_acs_cropped.dtype))

mosaic(kspace_acs_zero_filled, 4, 8, 1, [0,50], fig_size=(15,15), num_rot=0)
#%%
mosaic(kspace_acs_cropped, 4, 8, 1, [0,50], fig_size=(15,15), num_rot=0)

# %%
kspace_undersampled_zero_filled = np.zeros(kspace_fully_sampled.shape, dtype=kspace_fully_sampled.dtype)
kspace_undersampled_zero_filled[:,0:N1:Rx,:] = kspace_fully_sampled[:,0:N1:Rx,:]
# access every Rx line in the matrix 0:N1:Rx
# Visualize
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('kspace_undersampled_zero_filled Shape: ' + str(kspace_undersampled_zero_filled.shape))
print('kspace_undersampled_zero_filled Data Type: ' + str(kspace_undersampled_zero_filled.dtype))

mosaic(kspace_undersampled_zero_filled, 4, 8, 1, [0,50], fig_size=(15,15), num_rot=0)

# %%
print('Reconstruction Parameters')
print('---------------------------------------------')
print('Nx: ' + str(N1))
print('Ny: ' + str(N2))
print('ACS Size X: ' + str(num_acs[0]))
print('ACS Size Y: ' + str(num_acs[1]))
print('Rx: ' + str(Rx))
print()

print('ACS Data')
print('---------------------------------------------')
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('kspace_acs_zero_filled Shape: ' + str(kspace_acs_zero_filled.shape))
print('kspace_acs_zero_filled Data Type: ' + str(kspace_acs_zero_filled.dtype))
print()
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('kspace_acs_cropped Shape: ' + str(kspace_acs_cropped.shape))
print('kspace_acs_cropped Data Type: ' + str(kspace_acs_cropped.dtype))
print()

print('Sampled Data')
print('---------------------------------------------')
print('Axis Labels: ' + '[num_chan, N1, N2]')
print('kspace_undersampled_zero_filled Shape: ' + str(kspace_undersampled_zero_filled.shape))
print('kspace_undersampled_zero_filled Data Type: ' + str(kspace_undersampled_zero_filled.dtype))

# %%
grappa_shift_x = np.zeros(num_chan, dtype=int)
grappa_shift_y = np.zeros(num_chan, dtype=int)

# %%
grappa_coil_images_without_sub = np.zeros(kspace_undersampled_zero_filled.shape, kspace_undersampled_zero_filled.dtype)
grappa_coil_kspaces_without_sub = np.zeros(kspace_undersampled_zero_filled.shape, dtype=kspace_undersampled_zero_filled.dtype)

grappa_coil_kspaces_without_sub, grappa_coil_images_without_sub = grappa(kspace_undersampled_zero_filled,kspace_acs_zero_filled, 
                                                                         Rx, 1, num_acs, grappa_shift_x, grappa_shift_y,kernel_size=np.array([3, 13]), lambda_tik=0)

# %%
acs_mask = np.zeros((num_chan, N1, N2), dtype=bool)
acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = True

grappa_coil_kspaces_sub = (grappa_coil_kspaces_without_sub * np.invert(acs_mask).astype(int)) + kspace_acs_zero_filled[:,:,:]

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

comparison_grappa = np.concatenate((np.expand_dims(image_original, axis=0), np.expand_dims(grappa_rssq_image_sub, axis=0)), axis = 0)
rmse_grappa_sub = np.sqrt(np.sum(np.square(image_original - grappa_rssq_image_sub))) / np.sqrt(np.sum(np.square(image_original)))

print('rmse_grappa: ' + str(rmse_grappa_sub))
print()
print('Original Image, Grappa Image')
mosaic(comparison_grappa , 1, 2, 1, [0,1], fig_size=(15,15), num_rot=0)

# %%

acs_mask = np.zeros((num_chan, N1, N2), dtype=bool)
acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = True

kspace = kspace_undersampled_zero_filled * np.invert(acs_mask).astype(int) + kspace_acs_zero_filled[:,:,:]
kspace = np.rot90(kspace,k=1,axes=(1,2))
mosaic(kspace, 4, 8, 1, [0,50], fig_size=(15,15), num_rot=0)

kspace = np.transpose(kspace,(1,2,0))
print(kspace.shape)

dt = {'kspace' : kspace}
sio.savemat('rawdata', dt)

# %%
def weight_variable(shape,vari_name):                   
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])

#### LEARNING FUNCTION ####
def learning(ACS_input,target_input,accrate_input,sess):
    
    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                  
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z])         
    
    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)

    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1') 
    h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input)) 

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2,accrate_input))

    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3,accrate_input)

    error_norm = tf.norm(input_Target - h_conv3)       
    train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1 
    for i in range(MaxIteration+1):
        
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target})
        if i % 100 == 0:                                                                      
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})    
            print('The',i,'th iteration gives an error',error_now)                             
            
            
        
    error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    return [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),error]  


def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,acc_rate,sess):                
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate) 
    return sess.run(h_conv3)      

# %%
#### Network Parameters ####
kernel_x_1 = 3
kernel_y_1 = 2

kernel_x_2 = 1
kernel_y_2 = 1

kernel_last_x = 3
kernel_last_y = 2

layer1_channels = 32 
layer2_channels = 8

MaxIteration = 1000
LearningRate = 3e-3
#%%
#### Input/Output Data ####
inputData = 'rawdata.mat'
input_variable_name = 'kspace'
resultName = 'RAKI_recon'
recon_variable_name = 'kspace_recon'

# %%
# Read data
kspace = sio.loadmat(inputData)
kspace = kspace[input_variable_name] 
no_ACS_flag = 0
#%%
# ACS = sio.loadmat(inputData)
# ACS_data = ACS[input_variable_name]
# ACS_data = np.rot90(ACS_data,k = 1,axes =(1,2))
# ACS_data = np.transpose(ACS_data,(1,2,0))
# #%%
# kspace_undersampled = sio.loadmat('img_data_meas1_slice5.mat')
# kspace_us_data = kspace_undersampled['img_data_meas1_slice5']
# kspace_us_data = np.rot90(kspace_us_data,k=1,axes=(1,2))
# kspace_us_data = np.transpose(kspace_us_data,(1,2,0))
#%%
# Normalization
normalize = 0.015/np.max(abs(kspace[:]))
kspace = np.multiply(kspace,normalize)   
# normalize = 0.015/np.max(abs(kspace_us_data[:]))
# kspace_us_data = np.multiply(kspace_us_data,normalize)   

# Get the shapes
[m1,n1,no_ch] = np.shape(kspace)
# [m1,n1,no_ch] = np.shape(kspace_us_data)
no_inds = 1
kspace_all = kspace
# kspace_all = kspace_us_data

# %%
# Kspace grid
kx = np.transpose(np.int32([(range(1,m1+1))]))                  
ky = np.int32([(range(1,n1+1))])

# %%
# kspace = np.copy(kspace_all)
# mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0; 
# picks = np.where(mask == 1); 
kspace = np.copy(kspace_all)
mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace_all),0),1))>0; 
picks = np.where(mask == 1); 

# %%
# Cut the leftmost zero entries in kspace?
kspace = kspace[:,np.int32(picks[0][0]):n1+1,:] # n1+1 is out of index by 1
kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]

# %%
# Determine ACS Indices
kspace_NEVER_TOUCH = np.copy(kspace_all)

mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0;  
picks = np.where(mask == 1);                                  
d_picks = np.diff(picks,1)  
indic = np.where(d_picks == 1)

# %%
# Get x indices where there is data
mask_x = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),2),1))>0;
picks_x = np.where(mask_x == 1)
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

# %%
# Parse the ACS

if np.size(indic)==0:
    scaling = 1    
    no_ACS_flag=1;       
    print('No ACS signal in input data, using individual ACS file.')
    # matfn = 'ACS.mat'   
    # ACS = sio.loadmat(matfn)
    # ACS = ACS['ACS']  
    ACS = ACS_data   
    normalize = 0.015/np.max(abs(ACS[:])) 
    ACS = np.multiply(ACS,normalize*scaling) # There is no variable declaration for scaling?!

    kspace = np.multiply(kspace,scaling)
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
else:
    no_ACS_flag=0
    print('ACS signal found in the input data')
    indic = indic[1][:]
    center_start = picks[0][indic[0]]
    center_end = picks[0][indic[-1]+1]; # There is an indexing mistake here; center_end = picks[0][indic[-1]]
    ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)


# %%
# Get the acceleration rate

acc_rate = d_picks[0][0]
no_channels = ACS_dim_Z*2



# %%
# Pre-allocation

name_weight = resultName + ('_weight_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))
name_image = resultName + ('_image_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))

existFlag = os.path.isfile(name_image)

w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels],dtype=np.float32)
w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)
w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,acc_rate - 1, no_channels],dtype=np.float32)    

# What the !?
b1_flag = 0;
b2_flag = 0;                       
b3_flag = 0;

if (b1_flag == 1):
    b1_all = np.zeros([1,1, layer1_channels,no_channels]);
else:
    b1 = []

if (b2_flag == 1):
    b2_all = np.zeros([1,1, layer2_channels,no_channels])
else:
    b2 = []

if (b3_flag == 1):
    b3_all = np.zeros([1,1, layer3_channels, no_channels])
else:
    b3 = []

# %%
# ACS limits so that all kernels stay in the ACS region

target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1); 
target_x_end = np.int32(ACS_dim_X - target_x_start -1); 

time_ALL_start = time.time()

[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) 
ACS = np.float32(ACS)  

target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate;     
target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1;

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = acc_rate - 1

# %%
print('go!')
time_Learn_start = time.time() 

errorSum = 0;
config = tf.ConfigProto()


for ind_c in range(ACS_dim_Z):

    sess = tf.Session(config=config)
    # set target lines
    target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
    print('learning channel #',ind_c+1)
    time_channel_start = time.time()
    
    for ind_acc in range(acc_rate-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
        target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

    # learning

    [w1,w2,w3,error]=learning(ACS,target,acc_rate,sess) 
    w1_all[:,:,:,:,ind_c] = w1
    w2_all[:,:,:,:,ind_c] = w2
    w3_all[:,:,:,:,ind_c] = w3                               
    time_channel_end = time.time()
    print('Time Cost:',time_channel_end-time_channel_start,'s')
    print('Norm of Error = ',error)
    errorSum = errorSum + error

    sess.close()
    tf.reset_default_graph()
    
time_Learn_end = time.time();
print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min')
sio.savemat(name_weight, {'w1': w1_all,'w2': w2_all,'w3': w3_all})  

# %%
kspace_recon_all = np.copy(kspace_all)
kspace_recon_all_nocenter = np.copy(kspace_all)

kspace = np.copy(kspace_all)

# Find oversampled lines and set them to zero
over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
kspace_und = kspace
kspace_und[:,over_samp,:] = 0;
[dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

# Split real and imaginary parts
kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
kspace_und_re = np.float32(kspace_und_re)
kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_recon = kspace_und_re

# %%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/3 ; 

for ind_c in range(0,no_channels):
    print('Reconstruting Channel #',ind_c+1)
    
    sess = tf.Session(config=config) 
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    
    # grab w and b
    w1 = np.float32(w1_all[:,:,:,:,ind_c])
    w2 = np.float32(w2_all[:,:,:,:,ind_c])     
    w3 = np.float32(w3_all[:,:,:,:,ind_c])

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];                
        
    res = cnn_3layer(kspace_und_re,w1,b1,w2,b2,w3,b3,acc_rate,sess) 
    target_x_end_kspace = dim_kspaceUnd_X - target_x_start;
    
    for ind_acc in range(0,acc_rate-1):

        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1;             
        target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc;
        kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = res[0,:,::acc_rate,ind_acc]

    sess.close()
    tf.reset_default_graph()
    
kspace_recon = np.squeeze(kspace_recon)

kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); 


if no_ACS_flag == 0: 
    kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
    print('ACS signal has been putted back')
else:
    print('No ACS signal is putted into k-space')

kspace_recon_all[:,:,:] = kspace_recon_complex; 

for sli in range(0,no_ch):
    kspace_recon_all[:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,sli])

rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
sio.savemat(name_image,{recon_variable_name:kspace_recon_complex})

time_ALL_end = time.time()
print('All process costs ',(time_ALL_end-time_ALL_start)/60,'mins')
print('Error Average in Training is ',errorSum/no_channels)

# %%
dt = sio.loadmat('RAKI_recon_image_52,11,32_32,8.mat')

kspace_recon = dt['kspace_recon']
kspace_recon = np.transpose(kspace_recon,(2,0,1))
kspace_recon = np.rot90(kspace_recon,k=-1,axes=(1,2))
kspace_recon = kspace_recon/normalize

mosaic(kspace_recon, 4, 8, 1, [0,50], fig_size=(15,15), num_rot=0)

# %%
coil_images_raki_sub = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))

# Visualize
print('Axis Labels: ' + '[num_coils, N1, N2]')
print('RAKI Coil Images Shape: ' + str(coil_images_raki_sub.shape))
print('RAKI Coil Images Data Type: ' + str(coil_images_raki_sub.dtype))

raki_rssq_image_sub = np.sqrt(np.sum(np.square(np.abs(coil_images_raki_sub)),axis=0))

comparison_raki = np.concatenate((np.expand_dims(image_original, axis=0), np.expand_dims(raki_rssq_image_sub, axis=0)), axis = 0)
rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)))) / np.sqrt(np.sum(np.square(np.abs(image_original))))

print('rmse_raki: ' + str(rmse_raki_sub))
print()
print('Original Image, Rakı Image')
mosaic(comparison_raki , 1, 2, 1, [0,1], fig_size=(15,15), num_rot=0)

# %%
