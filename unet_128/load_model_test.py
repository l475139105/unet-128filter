#载入训练好的网络权重，
#从网络输出label图片。查看结果
#结果包括：损失函数 准确度 分割后的图片识别结果
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt


def load_patch_test_images(num_split_image=1):#获取一张 test集 数据。输出图像用
    patches = np.empty((num_split_image, 64, 64))
    for i in range(num_split_image):
        # train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_split_image/s%d.png'%i, 0)
        train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_split_image/s19377.png', 0)

        patches[i] = train_image_patch
    # patches = pre_process.my_PreProc(patches)#图像增强：自适应直方平均 归一化 adjust_gamma
    patches = np.expand_dims(patches,axis=3)
    patches = np.transpose(patches,(0,3,1,2))
    return patches.astype(np.uint8)
def load_patch_test_label(num_split_image=1):#获取一张 test 标签
    patches = np.empty((num_split_image, 64, 64))
    #print(patches.shape)
    for i in range(num_split_image):
        label_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_spilt_label/s3.png', 0)
        patches[i] = label_image_patch
    patches = patches.reshape(num_split_image,64*64)
    patches = patches.reshape([-1])
    patches[patches==255] =1
    patches = patches.reshape([num_split_image, 64 * 64, 1])
    return patches.astype(np.uint8)
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
def load_patch_train_images(num_split_image=5):#评估用
    patches = np.empty((num_split_image, 64, 64))
    for i in range(num_split_image):
        train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_split/s%d.png'%i, 0)

        patches[i] = train_image_patch
    # patches = pre_process.my_PreProc(patches)#图像增强：自适应直方平均 归一化 adjust_gamma
    patches = np.expand_dims(patches,axis=3)
    patches = np.transpose(patches,(0,3,1,2))
    return patches.astype(np.uint8)

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
    return patches.astype(np.uint8)


n_ch = 1
patch_height = 64
patch_width = 64
choose_split_image_num = 20000#一共1855个切割后的图片
model = get_unet(n_ch, patch_height, patch_width)
model.load_weights('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_last_weights.h5')
train_split_image = load_patch_train_images(choose_split_image_num)#一共1855个切割后的图片
lebel_split_image = load_patch_label_images(choose_split_image_num)
real_input_split_image = int(choose_split_image_num * 0.997)
x_test = train_split_image[real_input_split_image + 1:]#验证集
y_test = lebel_split_image[real_input_split_image + 1:]


def show_time(img):
    win = cv2.namedWindow('test win',cv2.WINDOW_NORMAL)
    # print(img1)
    cv2.imshow('test win', img)
    cv2.waitKey(3000)
    cv2.destroyWindow('test win')

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('Loss: ', loss)
print('Accuracy: ', acc)
pre_image = model.predict_on_batch(load_patch_test_images(1))#predict_generator
pre_image = np.argmax(pre_image,axis=2)
pre_image = pre_image.reshape(64,64)
np.set_printoptions(threshold=np.inf)
print(pre_image)

plt.figure()
plt.imshow(pre_image)
plt.show()

# show_time(pre_image)
# real_image = load_patch_test_label(1)
# real_image = real_image.reshape(64,64)
# print(real_image)
# show_time(real_image)