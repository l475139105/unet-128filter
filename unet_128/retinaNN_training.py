#训练网络
#20000个64*64的小图
#训练200遍
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,array_to_img
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
import tensorflow as tf
from keras.utils import  to_categorical
from six.moves import xrange

from keras.utils import generic_utils
import os
from scipy import  ndimage
import sys



# tensorflow WHC theno CWH
# 指定参数
# rotation_range 旋转
# width_shift_range 左右平移
# height_shift_range 上下平移
# zoom_range 随机放大或缩小
img_generator = ImageDataGenerator(
    rotation_range = 0,
    width_shift_range = 0,
    height_shift_range = 0,
    zoom_range = 0
    )


def load_patch_train_images(num_split_image=5):
    patches = np.empty((num_split_image, 64, 64))
    for i in range(num_split_image):
        train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_split/s%d.png'%i, 0)
        patches[i] = train_image_patch
    # patches = pre_process.my_PreProc(patches)#图像增强：自适应直方平均 归一化 adjust_gamma
    patches = np.expand_dims(patches,axis=3)
    patches = np.transpose(patches,(0,3,1,2))
    print(patches.shape)
    return patches.astype(np.uint8)#100 1 64 64

def load_patch_label_images(num_split_image=5):
    patches = np.empty((num_split_image, 64, 64))
    #print(patches.shape)
    for i in range(num_split_image):
        label_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Label_split/s%d.png'%i, 0)
        patches[i] = label_image_patch
    patches = patches.reshape(num_split_image,64*64)
    patches = patches.reshape([-1])
    patches[patches==255] =1
    patches = patches.reshape([num_split_image, 64 * 64, 1])

    return patches.astype(np.uint8)#100,4096,1


#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height * patch_width))(conv6)

    # print(conv6.shape)
    conv6 = core.Permute((2, 1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)
    # print(conv7.shape)
    model = Model(inputs=inputs, outputs=conv7)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#========= Load settings from Config file

# img = load_patch_train_images(1)
def show_time(img):
    win = cv2.namedWindow('test win',cv2.WINDOW_NORMAL)
    #显示图片
    img1 = img[0]
    print(img1)
    cv2.imshow('test win', img1)
    #设置图片的显示时间
    cv2.waitKey(3000)
    # #关闭图片窗口   ()里面输入你需要关闭的窗口
    cv2.destroyWindow('test win')


#=========== Construct and save the model arcitecture =====
n_ch = 1
patch_height = 64
patch_width = 64
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
model.load_weights('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5')

epochs = 40
batch_size = 10
choose_split_image_num = 5000#一共1855个切割后的图片
#============  Training ==================================
#方法一
# checkpointer = ModelCheckpoint(filepath='/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
# model.fit(load_patch_train_images(choose_split_image_num), load_patch_label_images(choose_split_image_num), epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

## 方法二
train_split_image = load_patch_train_images(choose_split_image_num)#一共1855个切割后的图片
lebel_split_image = load_patch_label_images(choose_split_image_num)

# real_input_split_image = int(choose_split_image_num * 0.95)#方法二用
# x_train_part = train_split_image[:real_input_split_image]#训练集
# y_train_part = lebel_split_image[:real_input_split_image]
# x_test = train_split_image[real_input_split_image + 1:]#验证集
# y_test = lebel_split_image[real_input_split_image + 1:]

x_test = train_split_image[19950:]#验证集 方法三用
y_test = lebel_split_image[19950:]
train_split_image = 0
lebel_split_image = 0

#方法二
# for e in range(epochs*1):
#     print('Epoch', e)
#     print('Training...')
#     progbar = generic_utils.Progbar(x_train_part.shape[0])
#     batches = 0
#
#     for x_batch, y_batch in img_generator.flow(x_train_part, y_train_part, batch_size=batch_size, shuffle=True):
#         loss,train_acc = model.train_on_batch(x_batch, y_batch)
#         # show_time(x_batch[0])
#         #print(x_batch.shape)  大小为 (batch_size , 1, 64,64)
#         batches += x_batch.shape[0]
#         if batches > x_train_part.shape[0]:
#             break
#         progbar.add(x_batch.shape[0], values=[('train loss', loss),('train acc', train_acc)])

#方法三
# for e in range(epochs*4):
#     print('Epoch', e)
#     print('Training...')


    # batches = 0
for e in range(epochs):
    for i in range(200):
        if i%100 == 0:
            model.save_weights('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5', overwrite=True)
        print('Epoch', e)
        print(i,'/200')
        batches = 0
        progbar = generic_utils.Progbar(100)
        read_label_image = np.memmap('/home/lyh/PycharmProjects/MachineLearn/unet_128/save_split_file/split_label_image_%d'%i, dtype='uint8', mode='r', shape=(100, 4096, 1))
        read_train_image = np.memmap('/home/lyh/PycharmProjects/MachineLearn/unet_128/save_split_file/split_train_image_%d'%i, dtype='uint8', mode='r', shape=(100,1,64, 64))
        print(read_train_image[0])
        for x_batch, y_batch in img_generator.flow(read_train_image, read_label_image, batch_size=batch_size, shuffle=True):
            loss, train_acc = model.train_on_batch(x_batch, y_batch)
            batches += x_batch.shape[0]
            if batches > read_train_image.shape[0]:
                break
            progbar.add(x_batch.shape[0], values=[('train loss', loss), ('train acc', train_acc)])

    # for x_batch, y_batch in img_generator.flow(x_train_part, y_train_part, batch_size=batch_size, shuffle=True):
    #     loss,train_acc = model.train_on_batch(x_batch, y_batch)
    #     # show_time(x_batch[0])
    #     #print(x_batch.shape)  大小为 (batch_size , 1, 64,64)
    #     batches += x_batch.shape[0]
    #     if batches > x_train_part.shape[0]:
    #         break
    #     progbar.add(100, values=[('train loss', loss),('train acc', train_acc)])

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('Loss: ', loss)
print('Accuracy: ', acc)
#========== Save and test the last model ===================
model.save_weights('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5', overwrite=True)
#model.load_weights('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5')



