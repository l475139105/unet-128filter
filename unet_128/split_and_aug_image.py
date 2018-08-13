#把原图512*512 分割成 64*64的小图
#利用旋转镜像等，扩大数据集
import cv2
import numpy as np
import crop
def split_train_image(default_image = 10):
    l=0#分割后图片编号
    for i in range(default_image):#分割i张原始图片
        train = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_image/%d.png'%i,0 )
        for j in range(15):
            for k in range(15):
                cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_split/s%d.png'%l, train[32*j:32*j+64, 32*k:32*k+64])
                l+=1
        # plt.imshow(train)
        # plt.show()
    return l
def split_train_label(default_image = 10):
    l = 0
    for i in range(default_image):#分割i张原始图片
        train = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Label_image/%d.png'%i,0 )
        for j in range(15):
            for k in range(15):
                cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/Label_split/s%d.png'%l, train[32*j:32*j+64, 32*k:32*k+64])
                l+=1
    return  l
def load_patch_train_images(num_split_image=5):
    patches = np.empty((num_split_image, 64, 64))
    for i in range(num_split_image):
        train_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_split/s%d.png'%i, 0)
        patches[i] = train_image_patch
    return patches.astype(np.uint8)

def load_patch_label_images(num_split_image=5):
    patches = np.empty((num_split_image, 64, 64))
    #print(patches.shape)
    for i in range(num_split_image):
        label_image_patch = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Label_split/s%d.png'%i, 0)
        patches[i] = label_image_patch
    return patches.astype(np.uint8)

def aug_existing_image(images,labels,num):
    index = 0
    for image in images:
        creat_image , creat_label = crop.data_augment(image,labels[index])
        print(creat_image.shape)
        cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_split/s%d.png' % num,creat_image)
        cv2.imwrite('/home/lyh/PycharmProjects/MachineLearn/unet_128/Label_split/s%d.png' % num,creat_label)
        num+=1
        index+=1
num_of_image = split_train_image(29)
num_of_label = split_train_label(29)
#增加图片
# creat_from_image_num = 6524
# array = np.random.randint(1,creat_from_image_num,13476)
# images = load_patch_train_images(creat_from_image_num)
# labels = load_patch_label_images(creat_from_image_num)
#
# aug_existing_image(images[array],labels[array],num_of_image)
# print(images[array].shape)
# print("**************")
# train = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/Train_split/s0.png',0) # 读取和代码处于同一目录下的 lena.png
# np.set_printoptions(threshold=np.inf)
# print(train.shape)
# win = cv2.namedWindow('test win',cv2.WINDOW_NORMAL)
# #显示图片
# cv2.imshow('test win', train)
# #设置图片的显示时间
# cv2.waitKey(1000)
# # #关闭图片窗口   ()里面输入你需要关闭的窗口
# cv2.destroyWindow('test win')

