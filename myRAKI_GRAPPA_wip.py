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
#%%
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
recon_data_mb_calib = sio.loadmat('recon_raw_mb_calib.mat')
recon_data_mb_calib = recon_data_mb_calib['recon_raw_mb']
R = 2
scaling = 1/np.max(np.abs(recon_data_mb_calib[:]))
recon_data_mb_calib_scaled = np.multiply(scaling, recon_data_mb_calib)
# recon_data_us = np.copy(recon_data_mb_calib_scaled)
# recon_data_us[:,:,1::R,:] = 0
ACS = recon_data_mb_calib_scaled[:,:,59-29:59+30,:]
ACS = np.transpose(ACS,[1,2,0,3])
[NCol, NRow, NCha, NSlc] = ACS.shape
#%%
ACS_slc4 = ACS[:,:,:,3]
ACS_slc4_re = np.zeros([NCol, NRow, NCha*2])
ACS_slc4_re[:,:,0:32] = ACS_slc4.real
ACS_slc4_re[:,:,32::] = ACS_slc4.imag
#%%
target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1)
target_x_end = np.int32(NCol - target_x_start -1)
target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * R;     
target_y_end = NRow  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * R -1
target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = R - 1
ACS_slc4_re = np.expand_dims(ACS_slc4_re,0)
#%%
input_data = Input(shape=(NCol,NRow,NCha*2))
recon_model = Model(input_data, RAKI(input_data,R))
recon_model.compile(loss='mae',optimizer=Adam())
recon_model.summary()
recon_model.save_weights('initial_weights_grappa.h5')
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
for ch in range(NCha*2):
    target = np.zeros([1,target_dim_X, target_dim_Y, target_dim_Z])
    print('learning channel #', ch+1)
    for ind_acc in range(R-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * R + ind_acc + 1 
        target_y_end = NRow  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * R + ind_acc
        target[0,:,:,ind_acc] = ACS_slc4_re[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ch]
    filename='weights_grappa/grappa_recon_model_'+str(ch)+'.h5'
    recon_model.load_weights('initial_weights_grappa.h5')
    recon_model_train = recon_model.fit(ACS_slc4_re, target, epochs=100,verbose=1,batch_size=1)
    recon_model.save_weights(filename)
#%%
slc = 3
kspace_recon_all = np.copy(recon_data_mb_calib_scaled)
kspace_recon_us = kspace_recon_all[:,:,:,slc]
kspace_recon_us[:,:,1::R] = 0
#%%
kspace_recon_us = np.transpose(kspace_recon_us,[1,2,0])
[NCol,NRow,NCha] = kspace_recon_us.shape
kspace_recon_us_re = np.zeros([NCol, NRow, NCha*2])
kspace_recon_us_re[:,:,0:NCha] = kspace_recon_us.real
kspace_recon_us_re[:,:,NCha::] = kspace_recon_us.imag
kspace_recon_us_re = np.expand_dims(kspace_recon_us_re,0)
kspace_recon = kspace_recon_us_re 
#%%
for ch in range(NCha*2):
    print('Reconstructing channel #' + str(ch+1))
    filename='weights_grappa/grappa_recon_model_'+str(ch)+'.h5'
    input_data_valid = Input(shape=(NCol,NRow,NCha*2))
    recon_model_valid = Model(input_data_valid, RAKI(input_data_valid,R))
    recon_model_valid.load_weights(filename)
    res = recon_model_valid.predict(kspace_recon_us_re)
    target_x_end_kspace = NCol - target_x_start
    for ind_acc in range(0,R-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * R + ind_acc + 1
        target_y_end_kspace = NRow - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * R + ind_acc
        kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:R,ch] = res[0,:,::R,ind_acc]
#%%
kspace_recon_complex = np.zeros([NCol, NRow, NCha], dtype =complex)
kspace_recon_complex.real = kspace_recon[:,:,:,0:NCha]
kspace_recon_complex.imag = kspace_recon[:,:,:,NCha::]
#%%
# input_data_placeholder = Input(shape= (input_data_slc4_full.shape[0],input_data_slc4_full.shape[1],input_data_slc4_full.shape[2]))
# recon_grappa_model = Model(input_data_placeholder,RAKI2(input = input_data_placeholder,R=2))
# #%%
# recon_grappa_model.compile(loss = 'mae',optimizer = Adam())
# recon_grappa_model.summary()
# #%%
# recon_grappa_model.save_weights('grappa_initial_weights.h5')
# # %%
# for ch in range(target_data_slc4_full.shape[2]):
#     target_data_tensor = np.expand_dims(target_data_slc4_full[:,:,ch],0)
#     target_data_tensor = np.expand_dims(target_data_tensor,3)
    
#     recon_grappa_model.load_weights('grappa_initial_weights.h5')
#     grappa_model_train = recon_grappa_model.fit(input_data_slc4_full, target_data_tensor,batch_size=1,epochs=100,verbose=1)
#     filename='weights_grappa/grappa_'+str(ch)+'_'+str(4)+'.h5'
#     recon_grappa_model.save_weights(filename)

# # %%
# valid_input_data = np.squeeze(recon_data_us[:,:,0::R,:])
# valid_target_data = np.squeeze(recon_data_mb_calib_scaled[:,:,1::R,:])
# #%%
# valid_input_data_full = np.zeros([valid_input_data.shape[1],valid_input_data.shape[2],valid_input_data.shape[0]*2,valid_input_data.shape[3]])
# valid_input_data_full[:,:,0:32,:] = np.transpose(valid_input_data.real,[1,2,0,3])
# valid_input_data_full[:,:,32::,:] = np.transpose(valid_input_data.imag,[1,2,0,3])
# #%%
# valid_target_data_full = np.zeros([valid_target_data.shape[1]-2,valid_target_data.shape[2],valid_target_data.shape[0]*2,valid_target_data.shape[3]])
# valid_target_data_full[:,:,0:32,:] = np.transpose(valid_target_data[:,1:valid_target_data.shape[1]-1,:,:].real,[1,2,0,3])
# valid_target_data_full[:,:,32::,:] = np.transpose(valid_target_data[:,1:valid_target_data.shape[1]-1,:,:].imag,[1,2,0,3])

# #%%
# valid_input = Input(shape=valid_input_data_full.shape[0:3])
# recon_valid_model = Model(valid_input, RAKI2(valid_input,2))
# recon_valid_model.compile(loss = 'mae',optimizer=Adam())

# # %%
# slc = 3
# valid_result = np.zeros(valid_target_data_full[:,:,:,3].shape)
# for ch in range(valid_input_data_full.shape[2]):
#     valid_input_tensor = np.expand_dims(valid_input_data_full[:,:,:,slc],0)
#     # valid_input_tensor = np.expand_dims(valid_input_tensor,3)
#     valid_target_tensor = np.expand_dims(valid_target_data_full[:,:,ch,slc],0)
#     valid_target_tensor = np.expand_dims(valid_target_tensor,3)
#     recon_valid_model.load_weights(filename)
#     valid_result[:,:,ch] = np.squeeze(recon_valid_model.predict(valid_input_tensor))


# # %%
# valid_padded = np.zeros([134,117,64])
# valid_padded[:,0::R,:] = valid_input_data_full[:,:,:,3]
# valid_padded[1:133,1::R,:] = valid_result


# # %%
# valid_padded_reshape = np.zeros([134,117,32],dtype=complex)
# valid_padded_reshape.real = valid_padded[:,:,0:32]
# valid_padded_reshape.imag = valid_padded[:,:,32::]

# # %%
