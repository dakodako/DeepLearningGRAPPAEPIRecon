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
#%%
R = 2
scaling = 1/np.max(np.abs(recon_data_mb_calib[:]))
recon_data_mb_calib_scaled = np.multiply(scaling, recon_data_mb_calib)
recon_data_us = np.copy(recon_data_mb_calib_scaled)
recon_data_us[:,:,0::R,:] = 0

ACS = recon_data_mb_calib_scaled[:,:,59-29:59+30,:]
target_data = ACS[:,:,1::R,:]
input_data = ACS[:,:,0::R,:]
target_data_slc4 = target_data[:,:,:,3]
input_data_slc4 = input_data[:,:,:,3]
target_data_slc4 = np.transpose(target_data_slc4,[1,2,0])
input_data_slc4 = np.transpose(input_data_slc4,[1,2,0])
# target_data_slc4_full = np.zeros([target_data_slc4.shape[0],target_data_slc4.shape[1],target_data_slc4.shape[2],target_data_slc4[3]*2])
target_data_slc4_full = np.zeros([target_data_slc4.shape[0]-2,target_data_slc4.shape[1], target_data_slc4.shape[2]*2])#, target_data_slc4.shape[3]*2])
target_data_slc4_full[:,:,0:32] = target_data_slc4[1:target_data_slc4.shape[0]-1,:,:].real
target_data_slc4_full[:,:,32::] = target_data_slc4[1:target_data_slc4.shape[0]-1,:,:].imag
input_data_slc4_full = np.zeros([input_data_slc4.shape[0], input_data_slc4.shape[1], input_data_slc4.shape[2]*2])#, input_data_slc4.shape[3]*2])
input_data_slc4_full[:,:,0:32] = input_data_slc4.real
input_data_slc4_full[:,:,32::] = input_data_slc4.imag
#%%
input_data_placeholder = Input(shape= (134,30,64))
recon_grappa_model = Model(input_data_placeholder,RAKI2(input = input_data_placeholder,R=2))
#%%
recon_grappa_model.compile(loss = 'mae',optimizer = Adam())
recon_grappa_model.summary()
#%%
recon_grappa_model.save_weights('grappa_initial_weights.h5')
#%%
target_data_slc4_full = np.expand_dims(target_data_slc4_full[:,:,0],0)
target_data_slc4_full = np.expand_dims(target_data_slc4_full,3)
input_data_slc4_full = np.expand_dims(input_data_slc4_full,0)
# input_data_slc4_full = np.expand_dims(input_data_slc4_full,3)
#%%
# scaling = 1/max(np.abs(target_data_slc4_full[:]))
# target_data_slc4_full_scaled = np.multiply(scaling, target_data_slc4_full)
# scaling = 1/max(np.abs(input_data_slc4_full[:]))
# input_data_slc4_full_scaled = np.multiply(scaling, input_data_slc4_full)
recon_grappa_model.load_weights('grappa_initial_weights.h5')
grappa_model_train = recon_grappa_model.fit(input_data_slc4_full,target_data_slc4_full,batch_size=1,epochs=100,verbose=1)
#%%
recon_grappa_valid = recon_grappa_model.predict(input_data_slc4_full)
#%%
recon_grappa_padded = np.zeros([134,59])
recon_grappa_padded[:,0::R] = input_data_slc4_full[0,:,:,0]
recon_grappa_padded[1:133,1::R] = np.squeeze(recon_grappa_valid)
#%%
# recon_model.save('.h5')
inputData = 'rawdata.mat'
input_variable_name = 'kspace'
resultName = 'RAKI_recon'
recon_variable_name = 'kspace_recon'
kspace = sio.loadmat(inputData)
kspace = kspace[input_variable_name] 
no_ACS_flag = 0
normalize = 0.015/np.max(abs(kspace[:]))
kspace = np.multiply(kspace,normalize) #%%
# target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1)
# target_x_end = np.int32(ACS_dim_X - target_x_start -1)
# %%
[m1,n1,no_ch] = np.shape(kspace)
no_inds = 1

kspace_all = kspace
kx = np.transpose(np.int32([(range(1,m1+1))]))                  
ky = np.int32([(range(1,n1+1))])

kspace = np.copy(kspace_all)
mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0
picks = np.where(mask == 1)                                
kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  

kspace_NEVER_TOUCH = np.copy(kspace_all)

mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0;  
picks = np.where(mask == 1)                                 
d_picks = np.diff(picks,1)  
indic = np.where(d_picks == 1)

mask_x = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),2),1))>0;
picks_x = np.where(mask_x == 1)
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

if np.size(indic)==0:    
    no_ACS_flag=1     
    print('No ACS signal in input data, using individual ACS file.')
    matfn = 'ACS.mat'   
    ACS = sio.loadmat(matfn)
    ACS = ACS['ACS']     
    normalize = 0.015/np.max(abs(ACS[:]))
    scaling = 1 
    ACS = np.multiply(ACS,normalize*scaling)

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
    center_end = picks[0][indic[-1]+1]
    ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)

acc_rate = d_picks[0][0]
no_channels = ACS_dim_Z*2

# %%
target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1)
target_x_end = np.int32(ACS_dim_X - target_x_start -1)

# time_ALL_start = time.time()

[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) 
ACS = np.float32(ACS)  

target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate;     
target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = acc_rate - 1

# %%
input_data = Input(shape=(ACS_dim_X,ACS_dim_Y,ACS_dim_Z))
recon_model = Model(input_data,RAKI(input_data,6))
recon_model.compile(loss='mae',optimizer=Adam())
recon_model.summary()
recon_model.save_weights('initial_weights_grappa.h5')
# %%
for ch in range(ACS_dim_Z):
    # sess = tf.Session(config=config)
    # set target lines
    target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
    print('learning channel #',ch+1)
    # time_channel_start = time.time()
    for ind_acc in range(acc_rate-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
        target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ch]
    filename = 'weights_grappa/' + 'grappa_recon_model_' + str(ch)+'.h5'
    recon_model.load_weights('initial_weights_grappa.h5')
    recon_model_train = recon_model.fit(ACS, target, epochs = 1, verbose = 1, batch_size = 1)
    recon_model.save_weights(filename)
# %%
kspace_recon_all = np.copy(kspace_all)
kspace_recon_all_nocenter = np.copy(kspace_all)

kspace = np.copy(kspace_all)

over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
kspace_und = kspace
kspace_und[:,over_samp,:] = 0
[dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
kspace_und_re = np.float32(kspace_und_re)
kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_recon = kspace_und_re
#%%
for ch in range(0,no_channels):
    print('Reconstruting Channel #',ch+1)
    
    # sess = tf.Session(config=config) 
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        # init = tf.initialize_all_variables()
    # else:
        # init = tf.global_variables_initializer()
    # sess.run(init)
    
    # grab w and b
    # w1 = np.float32(w1_all[:,:,:,:,ind_c])
    # w2 = np.float32(w2_all[:,:,:,:,ind_c])     
    # w3 = np.float32(w3_all[:,:,:,:,ind_c])

    # if (b1_flag == 1):
    #     b1 = b1_all[:,:,:,ind_c];
    # if (b2_flag == 1):
    #     b2 = b2_all[:,:,:,ind_c];
    # if (b3_flag == 1):
    #     b3 = b3_all[:,:,:,ind_c];
    filename = 'weights_grappa/' + 'grappa_recon_model_' + str(ch)+'.h5'
    input_data_valid = Input(shape = np.squeeze(kspace_und_re).shape)
    recon_model_valid = Model(input_data_valid,RAKI(input_data_valid, 6))
    # Model(input_data_mb, RAKI_SMS(input_data_mb))
    recon_model_valid.load_weights(filename)                
    res = recon_model_valid.predict(kspace_und_re)  
    # res = cnn_3layer(kspace_und_re,w1,b1,w2,b2,w3,b3,acc_rate,sess) 
    target_x_end_kspace = dim_kspaceUnd_X - target_x_start
    
    for ind_acc in range(0,acc_rate-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1
        target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc
        kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ch] = res[0,:,::acc_rate,ind_acc]

    # sess.close()
    # tf.reset_default_graph()
#%%   
kspace_recon = np.squeeze(kspace_recon)
kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex)

# %%
if no_ACS_flag == 0: 
    kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
    print('ACS signal has been putted back')
else:
    print('No ACS signal is putted into k-space')
# %%
kspace_recon_all[:,:,:] = kspace_recon_complex
# %%
