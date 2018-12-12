import random
import os
import cv2
import math
from tensorflow.examples.tutorials.mnist import input_data
PATH = os.path.dirname(os.path.abspath(__file__))
DIR=os.path.join(PATH,"data")
MDIR=os.path.join(DIR,"modelsave")
DDIR=os.path.join(DIR,"000")
LDIR=os.path.join(DDIR,"videos\lc\left1.mp4.frames")
RDIR=os.path.join(DDIR,"videos")
RDIR=os.path.join(RDIR,"rc")
RDIR=os.path.join(RDIR,"right1.mp4.frames")
SDIR=os.path.join(DDIR,"videos\sc\center1.mp4.frames")


'''
#shuffle验证
lista =list()
lista.append((20,1))
lista.append(( 16,2))
lista.append((10,3))
lista.append((5,4))
random.shuffle(lista)
print ("随机排序列表 : ",  lista)

random.shuffle(lista)
print( "随机排序列表 : ",  lista)

'''
'''
#图像归一化验证
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)


print('输入数据:', mnist.train.images)
print('输入数据打印shape:', mnist.train.images.shape)
import pylab
im = mnist.train.images[1]
im = im.reshape(-1, 28)
print(im)
pylab.imshow(im)
pylab.show()
print('输入数据打印shape:', mnist.test.images.shape)
print('输入数据打印shape:', mnist.validation.images.shape)
'''

#数据读取与增强
def read(dir):
    for _, __, file in os.walk(dir):
        print(file)  # 当前路径下所有非目录子文件
        files=file
    count = 0
    for i in files:

        CDIR = os.path.join(dir, i)

        img = cv2.imread(CDIR)
        img3=img[math.floor(img.shape[0]*15/100):math.floor(img.shape[0]*85/100), math.floor(img.shape[1]*15/100):math.floor(img.shape[1]*85/100)]
        print(img.shape)
        print(img3.shape)
        img = cv2.resize(img, (101, 101))
        img3=cv2.resize(img3,(101,101))
        print(img.shape)
        cv2.imshow('00000001.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img2 = cv2.flip(img, 1)
        print(img2.shape)
        cv2.imshow('00000001.jpg', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        #img3=img.astype(float)/255

        cv2.imshow('00000001.jpg', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#read(LDIR)



#截取帧
def get():
    print(DIR)
    name = os.path.join(DIR, "3.mp4")
    vc = cv2.VideoCapture(name)  # 读入视频文件
    c = 1

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
        print("success")
    else:
        rval = False
        print("fail")

    timeF = 25  # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作

            VDIR = os.path.join(DIR, "images2")
            if not os.path.exists(VDIR):
                os.mkdir(VDIR)
            cv2.imwrite(VDIR + '/' + str(c) + '.jpg', frame)  # 存储为图像
        c = c + 1
        cv2.waitKey(1)
    vc.release()



#图像输出打印
import matplotlib.pyplot as plt

num_list = [0.6, 0.3, 0.1]
plt.bar(range(-1,len(num_list)-1), num_list)
plt.show()