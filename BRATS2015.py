#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import random as r
from keras.models import Sequential,load_model,Model,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import SimpleITK as sitk
#K.set_image_data_format("channels_first")
K.set_image_dim_ordering("th")
img_size = 120      #original img size is 240*240
smooth = 1 
num_of_aug = 1
num_epoch = 20


#%%

import glob
def create_data(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    imgs = []
    print('Processing---', mask)
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        img = trans.resize(img, resize, mode='constant')
        if label:
            #img[img == 4] = 1       #turn enhancing tumor into necrosis
            #img[img != 1] = 0       #only left enhancing tumor + necrosis
            img[img != 0] = 1       #Region 1 => 1+2+3+4 complete tumor
            img = img.astype('float32')
        else:
            img = (img-img.mean()) / img.std()      #normalization => zero mean   !!!care for the std=0 problem
        for slice in range(50,130):
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
            img_g = augmentation(img_t,num_of_aug)
            for n in range(img_g.shape[0]):
                imgs.append(img_g[n,:,:,:])
    name = 'y_'+ str(img_size) if label else 'x_'+ str(img_size)
    np.save(name, np.array(imgs).astype('float32'))  # save at home
    print('Saved', len(files), 'to', name)

#%%

def n4itk(img):         #must input with sitk img object
    img = sitk.Cast(img, sitk.sitkFloat32)
    img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
    corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
    return corrected_img    

    
#%%

def augmentation(scans,n):          #input img must be rank 4 
    datagen = ImageDataGenerator(
        featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=25,   
        #width_shift_range=0.3,  
        #height_shift_range=0.3,   
        horizontal_flip=True,   
        vertical_flip=True,  
        zoom_range=False)
    i=0
    scans_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000): 
        scans_g=np.vstack([scans_g,batch])
        i += 1
        if i == n:
            break
    '''    remember arg + labels  
    i=0
    labels_g=labels.copy()
    for batch in datagen.flow(labels, batch_size=1, seed=1000): 
        labels_g=np.vstack([labels_g,batch])
        i += 1
        if i > n:
            break    
    return ((scans_g,labels_g))'''
    return scans_g
#scans_g,labels_g = augmentation(img,img1, 10)
#X_train = X_train.reshape(X_train.shape[0], 1, img_size, img_size)
    
#%%

'''
Model -

structure:

'''    

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
    
def unet_model():
    inputs = Input((1, img_size, img_size))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)      # KERNEL =3 STRIDE =3
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
    


    
#%%
# catch all T1c.mha
create_data('/home/andy/Brain_tumor/BRATS2015/BRATS2015_Training/HGG/', '**/*Flair*.mha', label=False, resize=(155,img_size,img_size))
create_data('/home/andy/Brain_tumor/BRATS2015/BRATS2015_Training/HGG/', '**/*OT*.mha', label=True, resize=(155,img_size,img_size))

#%%
# catch BRATS2017 Data
create_data('/home/andy/Brain_tumor/BRATS2017/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/', '**/*_flair.nii.gz', label=False, resize=(155,img_size,img_size))
create_data('/home/andy/Brain_tumor/BRATS2017/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/', '**/*_GlistrBoost_ManuallyCorrected.nii.gz', label=True, resize=(155,img_size,img_size))


#%%
# load numpy array data
x = np.load('/home/andy/x_{}.npy'.format(img_size))
y = np.load('/home/andy/y_{}.npy'.format(img_size))

#%%
#training
num = 31100

model = unet_model()
history = model.fit(x, y, batch_size=16, validation_split=0.2 ,nb_epoch= num_epoch, verbose=1, shuffle=True)
pred = model.predict(x[num:num+100])

#%%
# save model and weights
model.save('aug{}_{}_epoch{}'.format(num_of_aug,img_size,num_epoch))
model.save_weights('weights_{}_{}.h5'.format(img_size,num_epoch))
#model.load_weights('weights.h5')

#%%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
#show results
for n in range(2):
    i = int(r.random() * pred.shape[0])
    plt.figure(figsize=(15,10))

    plt.subplot(131)
    plt.title('Input'+str(i+num))
    plt.imshow(x[i+num, 0, :, :],cmap='gray')

    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(y[i+num, 0, :, :],cmap='gray')

    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(pred[i, 0, :, :],cmap='gray')

    plt.show()

#%%
'''
animation
'''
import matplotlib.animation as animation
def animate(pat, gifname):
    # Based on @Zombie's code
    fig = plt.figure()
    anim = plt.imshow(pat[50])
    def update(i):
        anim.set_array(pat[i])
        return anim,
    
    a = animation.FuncAnimation(fig, update, frames=range(len(pat)), interval=50, blit=True)
    a.save(gifname, writer='imagemagick')
    
#animate(pat, 'test.gif')
