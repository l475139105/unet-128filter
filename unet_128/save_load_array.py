#分割成N个文件，每次给数据送入一个文件



import numpy as np
import cv2

def load_patch_train_images(num_split_image=5):
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

num_image = 20000#想要读取的64 ×64 图片的数量
num_image_save_per_file = 100
train_split_image = load_patch_train_images(num_image)
print(train_split_image[4999])
for i in range(int(num_image/num_image_save_per_file)):#每个文件存储 100,1,64,64大小的
    mmap = np.memmap('/home/lyh/PycharmProjects/MachineLearn/unet_128/save_split_file/split_train_image_%d'%i, dtype='uint8', mode='w+', shape=(num_image_save_per_file, train_split_image.shape[1], train_split_image.shape[2], train_split_image.shape[3]))
    print('split_train_image', mmap.shape)
    mmap[:] = train_split_image[num_image_save_per_file * i:num_image_save_per_file * (i+1)]
    mmap.flush()

label_split_image = load_patch_label_images(num_image)
for i in range(int(num_image/num_image_save_per_file)):#每个文件存储 100,4096,1大小的
    mmap = np.memmap('/home/lyh/PycharmProjects/MachineLearn/unet_128/save_split_file/split_label_image_%d'%i, dtype='uint8', mode='w+',
                     shape=(num_image_save_per_file, label_split_image.shape[1], label_split_image.shape[2]))
    print('split_label_image', mmap.shape)
    mmap[:] = label_split_image[num_image_save_per_file * i:num_image_save_per_file * (i + 1)]
    mmap.flush()



read_train_image = np.memmap('/home/lyh/PycharmProjects/MachineLearn/unet_128/save_split_file/split_train_image_49', dtype='uint8', mode='r', shape=(100, 1,64,64))
print(read_train_image[99])
#







