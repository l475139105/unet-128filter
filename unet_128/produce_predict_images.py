#输出 合成后的预测大图512*512
#过程：把原图按照15*15的规格，重叠裁剪。原理与原论文中的镜像法差不多，消除合成图中的网格
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib


def split_train_image(default_image = 10):
    l=0#分割后图片编号
    for i in range(default_image):#分割i张原始图片
        # train = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_image/%d.png'%i,0 )#训练集
        train = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_images/6.png',0 )#测试集
        # #分割成 8 × 8
        # for j in range(8):
        #     for k in range(8):
        #         cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_split_image/s%d.png' % l,
        #                     train[64 * j:64 * j + 64, 64 * k:64 * k + 64])
        #         l+=1
        ##分割成  15 × 15
        for j in range(15):
            for k in range(15):
                cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_split_image/s%d.png'%l, train[32*j:32*j+64, 32*k:32*k+64])
                l+=1
        # plt.imshow(train)
        # plt.show()
    return l

def load_patch_test_images(num_split_image=1):#获取一张 test集 数据。输出图像用
    patches = np.empty((num_split_image, 64, 64))
    for i in range(num_split_image):
        # train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_split_image/s%d.png'%i, 0)
        train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_split_image/s%d.png'%i, 0)

        patches[i] = train_image_patch
    # patches = pre_process.my_PreProc(patches)#图像增强：自适应直方平均 归一化 adjust_gamma
    patches = np.expand_dims(patches,axis=3)
    patches = np.transpose(patches,(0,3,1,2))
    return patches.astype(np.uint8)
# def load_patch_test_label(num_split_image=1):#获取一张 test 标签
#     patches = np.empty((num_split_image, 64, 64))
#     #print(patches.shape)
#     for i in range(num_split_image):
#         label_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_spilt_label/s%d.png'%i, 0)
#         patches[i] = label_image_patch
#     patches = patches.reshape(num_split_image,64*64)
#     patches = patches.reshape([-1])
#     patches[patches==255] =1
#     patches = patches.reshape([num_split_image, 64 * 64, 1])
#     return patches.astype(np.uint8)
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


# split_images_num = split_train_image(10)#分割512×512 变成64*64
def recovery_patches(patches):
    recovery_patches = np.empty(( 512, 512))
    l = 0
    for j in range(15):
        print('.')
        for k in range(15):
            print(recovery_patches[32*j:32*j+64, 32*k:32*k+64].shape)
            recovery_patches[32*j:32*j+64, 32*k:32*k+64] = patches[l,:,:]
            l += 1

    l = 0
    for j in range(15):
        for k in range(15):
            recovery_patches[32*j+16  : 32*j+32+16  , 32*k+16:32*k+32+16] = patches[l,16:48,16:48]
            l += 1
    return recovery_patches


split_train_image(1)#只分割一张图片
n_ch = 1
patch_height = 64
patch_width = 64
choose_split_image_num = 225#一共 split_images_num 个切割后的图片
model = get_unet(n_ch, patch_height, patch_width)
model.load_weights('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5')
# train_split_image = load_patch_train_images(choose_split_image_num)#一共640后的图片
# print(train_split_image.shape)
pre_image = model.predict_on_batch(load_patch_test_images(choose_split_image_num))#predict_generator
print(pre_image.shape)
pre_image = np.argmax(pre_image,axis=2)
np.set_printoptions(threshold=np.inf)
print(pre_image.shape)

# fig ,ax_array = plt.subplots(nrows=1,ncols=5,sharey=True,sharex=True)
# ax_array[0].matshow(pre_image[0].reshape(64,64),cmap=matplotlib.cm.binary)
# ax_array[1].matshow(pre_image[1].reshape(64,64),cmap=matplotlib.cm.binary)
# ax_array[2].matshow(pre_image[2].reshape(64,64),cmap=matplotlib.cm.binary)
# ax_array[3].matshow(pre_image[3].reshape(64,64),cmap=matplotlib.cm.binary)
# ax_array[4].matshow(pre_image[4].reshape(64,64),cmap=matplotlib.cm.binary)
#
#
# plt.show()




pre_image = pre_image.reshape(225,64,64)
recvory_image = recovery_patches(pre_image)
# print(recvory_image[0])
plt.figure()
plt.imshow(recvory_image,cmap=matplotlib.cm.gray)
plt.show()

recvory_image[recvory_image==1]=255
#黑色 0  白 1   或
cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/6_pre_byL.png' ,recvory_image)


