#打印 预测与标签的报告
#因为标签是256*256 ，所以需要先把预测矩阵从512 缩小到256
import cv2
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# np.set_printoptions(threshold=np.inf)
label = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/test_spilt_label/6_predict.png' , 0)

label[label>=128] = 255
label[label<128] = 0
plt.figure()
plt.imshow(label,cmap=matplotlib.cm.gray)
plt.show()
label = label.reshape((256*256,1))
print(label)
recvory_image = cv2.imread('/home/lyh/PycharmProjects/MachineLearn/unet_128/6_pre_byL.png' , 0)
print('recvory_image',recvory_image.shape)
recvory_image = cv2.resize(recvory_image, (256,256), cv2.INTER_LINEAR)
print('recvory_image',recvory_image)
recvory_image = recvory_image.reshape((256*256,1))
print(classification_report(label, recvory_image))
