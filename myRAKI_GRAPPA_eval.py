#%%
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.layers import BatchNormalization as BatchNorm
from keras.models import Model, load_model
from keras.layers import Input, Reshape, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import layers
from keras.optimizers import Adam

#%%
import scipy.io as sio 
import numpy as np
import numpy.matlib
import h5py
#%%
R = 2
kernel_x_1 = 5
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
def RAKI(input,R):
    layer1_channels = 32
    layer2_channels = 8
    kernel_x_1 = 5
    kernel_y_1 = 2
    kernel_x_2 = 1
    kernel_y_2 = 1
    kernel_last_x = 3
    kernel_last_y = 2
    h_conv1 = Conv2D(layer1_channels, (kernel_x_1,kernel_y_1),activation = 'relu',padding='valid',dilation_rate=[1,R])(input)
    h_conv2 = Conv2D(layer2_channels, (kernel_x_2,kernel_y_2),activation = 'relu',padding='valid',dilation_rate=[1,R])(h_conv1)
    h_conv3 = Conv2D(R-1,(kernel_last_x,kernel_last_y),padding='valid',dilation_rate=[1,R])(h_conv2)
    return h_conv3
def RAKI2(input,R):
    layer1_channels = 32
    layer2_channels = 8
    kernel_x_1 = 5
    kernel_y_1 = 2
    kernel_x_2 = 1
    kernel_y_2 = 1
    kernel_last_x = 3
    kernel_last_y = 2
    h_conv1 = Conv2D(layer1_channels, (kernel_x_1,kernel_y_1),activation = 'relu',padding='same',dilation_rate=1)(input)
    h_conv2 = Conv2D(layer2_channels, (kernel_x_2,kernel_y_2),activation = 'relu',padding='same',dilation_rate=1)(h_conv1)
    h_conv3 = Conv2D(R-1,(kernel_last_x,kernel_last_y),padding='valid',dilation_rate=1)(h_conv2)
    return h_conv3
#%%
inputData = '../MRI_data/MRI_data/raw_data/raw_data_134_3107_PA/patrefscan_rmos_data_pc_corrected_reordered_134.mat'
input_variable_name = 'new_image_data'
arrays = {}
f = h5py.File(inputData)
for k,v in f.items():
    arrays[k] = np.array(v)
f.close()
#%%
data_full = np.zeros(arrays[input_variable_name].shape, dtype=complex)
data_full.real = arrays[input_variable_name]['real']
data_full.imag = arrays[input_variable_name]['imag']
#%%
data_full = np.transpose(data_full,[3,1,2,0])
data_full_padded = np.zeros((data_full.shape[0], data_full.shape[1]*2-1, data_full.shape[2], data_full.shape[3]),dtype = complex)
data_full_padded[:,0::R,:,:] = data_full
[NCol, data_NRow, NCha, NSlc] = data_full_padded.shape
R = 2
#%%

input_data_recon = Input(shape=(NCol, data_NRow, NCha*2))
recon_model_valid = Model(input_data_recon, RAKI(input_data_recon,R))
recon_model_valid.summary()
#%%
target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1)
target_x_end = np.int32(NCol - target_x_start -1)
target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * R     
target_y_end = data_NRow  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * R -1
target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = R - 1
#%%
kspace_recon_full = np.zeros((NCol, data_NRow, NCha*2,NSlc))
for slc in range(29,30):
    print('Slice ',slc+1)
    input_all = np.copy(data_full_padded[:,:,:,slc])
    scaling = 1/np.max(np.abs(input_all[:]))
    input_all_scaled = np.multiply(input_all,scaling)
    input_re = np.zeros([NCol, data_NRow, NCha*2])
    input_re[:,:,0:NCha] = input_all_scaled.real
    input_re[:,:,NCha::] = input_all_scaled.imag 
    input_re = np.expand_dims(input_re,0)
    kspace_recon = input_re
    
    for ch in range(NCha*2):
        print('Reconstructing channel ', ch+1)
        filename = 'weights_grappa_134_3107/grappa_recon_model_'+str(ch)+'_'+str(slc)+'.h5'
        recon_model_valid.load_weights(filename)
        res = recon_model_valid.predict(input_re) 
        # target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
        target_x_end_kspace = NCol - target_x_start
        for ind_acc in range(R-1):
            target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * R + ind_acc + 1
            target_y_end_kspace = data_NRow - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * R + ind_acc
            # kspace_recon_full[target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:R,ch,slc] = res[0,:,::R,ind_acc]
            kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:R,ch] = res[0,:,::R,ind_acc]
    kspace_recon_full[:,:,:,slc] = np.squeeze(kspace_recon)
    # kspace_recon_full[:,:,ch,slc] = np.squeeze(kspace_recon)
#%%
kspace_recon_full_complex = np.zeros((NCol, data_NRow, NCha, NSlc), dtype=complex) 
kspace_recon_full_complex.real = kspace_recon_full[:,:,0:NCha,:]
kspace_recon_full_complex.imag = kspace_recon_full[:,:,NCha::,:] 
#%%
# recon_data_mb_calib = sio.loadmat('recon_raw_mb_calib.mat')
# recon_data_mb_calib = recon_data_mb_calib['recon_raw_mb']
# R = 2
# scaling = 1/np.max(np.abs(recon_data_mb_calib[:]))
# recon_data_mb_calib_scaled = np.multiply(scaling, recon_data_mb_calib)
# recon_data_us = np.copy(recon_data_mb_calib_scaled)
# recon_data_us[:,:,1::R,:] = 0
# ACS = recon_data_mb_calib_scaled[:,:,59-29:59+30,:]
# ACS = np.transpose(ACS,[1,2,0,3])
# [NCol, NRow, NCha, NSlc] = ACS.shape
#%%
# ACS_slc4 = ACS[:,:,:,3]
# ACS_slc4_re = np.zeros([NCol, NRow, NCha*2])
# ACS_slc4_re[:,:,0:32] = ACS_slc4.real
# ACS_slc4_re[:,:,32::] = ACS_slc4.imag
#%%
# target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1)
# target_x_end = np.int32(NCol - target_x_start -1)
# target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * R
# target_y_end = NRow  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * R -1
# target_dim_X = target_x_end - target_x_start + 1
# target_dim_Y = target_y_end - target_y_start + 1
# target_dim_Z = R - 1
# ACS_slc4_re = np.expand_dims(ACS_slc4_re,0)
#%%
# input_data = Input(shape=(NCol,NRow,NCha*2))
# recon_model = Model(input_data, RAKI(input_data,R))
# recon_model.compile(loss='mae',optimizer=Adam())
# recon_model.summary()
# recon_model.save_weights('initial_weights_grappa.h5')
# target_data = ACS[:,:,1::R,:]
# input_data = ACS[:,:,0::R,:]
# target_data_slc4 = target_data[:,:,:,3]
# input_data_slc4 = input_data[:,:,:,3]
# target_data_slc4 = np.transpose(target_data_slc4,[1,2,0])
# input_data_slc4 = np.transpose(input_data_slc4,[1,2,0])
# # target_data_slc4_full = np.zeros([target_data_slc4.shape[0],target_data_slc4.shape[1],target_data_slc4.shape[2],target_data_slc4[3]*2])
# target_data_slc4_full = np.zeros([target_data_slc4.shape[0]-2,target_data_slc4.shape[1], target_data_slc4.shape[2]*2])#, target_data_slc4.shape[3]*2])
# target_data_slc4_full[:,:,0:32] = target_data_slc4[1:target_data_slc4.shape[0]-1,:,:].real
# target_data_slc4_full[:,:,32::] = target_data_slc4[1:target_data_slc4.shape[0]-1,:,:].imag
# input_data_slc4_full = np.zeros([input_data_slc4.shape[0], input_data_slc4.shape[1], input_data_slc4.shape[2]*2])#, input_data_slc4.shape[3]*2])
# input_data_slc4_full[:,:,0:32] = input_data_slc4.real
# input_data_slc4_full[:,:,32::] = input_data_slc4.imag
# input_data_slc4_full = np.expand_dims(input_data_slc4_full,0)
#%%
# for ch in range(NCha*2):
#     target = np.zeros([1,target_dim_X, target_dim_Y, target_dim_Z])
#     print('learning channel #', ch+1)
#     for ind_acc in range(R-1):
#         target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * R + ind_acc + 1 
#         target_y_end = NRow  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * R + ind_acc
#         target[0,:,:,ind_acc] = ACS_slc4_re[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ch]
#     filename='weights_grappa/grappa_recon_model_'+str(ch)+'.h5'
#     recon_model.load_weights('initial_weights_grappa.h5')
#     recon_model_train = recon_model.fit(ACS_slc4_re, target, epochs=100,verbose=1,batch_size=1)
#     recon_model.save_weights(filename)
#%%
# slc = 3
# kspace_recon_all = np.copy(recon_data_mb_calib_scaled)
# kspace_recon_us = kspace_recon_all[:,:,:,slc]
# kspace_recon_us[:,:,1::R] = 0
# #%%
# kspace_recon_us = np.transpose(kspace_recon_us,[1,2,0])
# [NCol,NRow,NCha] = kspace_recon_us.shape
# kspace_recon_us_re = np.zeros([NCol, NRow, NCha*2])
# kspace_recon_us_re[:,:,0:NCha] = kspace_recon_us.real
# kspace_recon_us_re[:,:,NCha::] = kspace_recon_us.imag
# kspace_recon_us_re = np.expand_dims(kspace_recon_us_re,0)
# kspace_recon = kspace_recon_us_re 
#%%
# for ch in range(NCha*2):
#     print('Reconstructing channel #' + str(ch+1))
#     filename='weights_grappa/grappa_recon_model_'+str(ch)+'.h5'
#     input_data_valid = Input(shape=(NCol,NRow,NCha*2))
#     recon_model_valid = Model(input_data_valid, RAKI(input_data_valid,R))
#     recon_model_valid.load_weights(filename)
#     res = recon_model_valid.predict(kspace_recon_us_re)
#     target_x_end_kspace = NCol - target_x_start
#     for ind_acc in range(0,R-1):
#         target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * R + ind_acc + 1
#         target_y_end_kspace = NRow - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * R + ind_acc
#         kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:R,ch] = res[0,:,::R,ind_acc]
#%%
# kspace_recon_complex = np.zeros([NCol, NRow, NCha], dtype =complex)
# kspace_recon_complex.real = kspace_recon[:,:,:,0:NCha]
# kspace_recon_complex.imag = kspace_recon[:,:,:,NCha::]