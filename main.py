import tensorflow as tf
import  numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import math
from memory import Memory
PATH = os.path.dirname(os.path.abspath(__file__))
DIR=os.path.join(PATH,"data")
MDIR=os.path.join(DIR,"modelsave")
DDIR=os.path.join(DIR,"000")
LDIR=os.path.join(DDIR,"videos\lc\left1.mp4.frames")
RDIR=os.path.join(DDIR,"videos")
RDIR=os.path.join(RDIR,"rc")
RDIR=os.path.join(RDIR,"right1.mp4.frames")
SDIR=os.path.join(DDIR,"videos\sc\center1.mp4.frames")

from model import Classifier


def main():
    iteration = 100001
    statistic_amount=200
    savecount=5000
    batch=256
    learning_rate=0.0002
    e = len([file for file in os.listdir(MDIR) if os.path.isdir(os.path.join(MDIR, file))])
    mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
    memory = Memory()
    getdata(memory)



    #print('输入数据:', mnist.train.images)
    #print('输入数据打印shape:', mnist.train.images.shape)
    #import pylab
    #im = mnist.train.images[1]
    #im = im.reshape(-1, 28)
    #pylab.imshow(im)
    #pylab.show()
    #print('输入数据打印shape:', mnist.test.images.shape)
    #print('输入数据打印shape:', mnist.validation.images.shape)

    with tf.device("/gpu:0"):
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.75  # 程序最多只能占用指定gpu50%的显存
        config.gpu_options.allow_growth=True


        with tf.Session(config=config) as sess:

            #global_episode=tf.Variable(0,dtype=tf.int32,trainable=False)

            classifier=Classifier(sess,3,learning_rate)
            tf.summary.scalar('acc', classifier.accuracy)
            tf.summary.scalar('loss', classifier.loss)
            summaryit = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir=DIR, graph=sess.graph)


            saver = tf.train.Saver()
            time = len([file for file in os.listdir(MDIR) if os.path.isdir(os.path.join(MDIR, file))]) -1
            sess.run(tf.global_variables_initializer())
            if time<0:
                print("no model")
                premodel=False
            elif classifier.load(saver, MDIR):
                print("have model")
            else:
                print("still have some problems")


            for i in range(0,iteration):
                X,Y=memory.sample(batch)
                classifier.caculate(X,Y,i)
                if i % statistic_amount==0:
                    C, D = memory.sampleVa()
                    train =classifier.statistic(C,D,i,summaryit,X,Y)
                    summary_writer.add_summary(train,i)
                if i % savecount == 0:
                    print("save------------------------")
                    nDir = os.path.join(MDIR, str(e))
                    if not os.path.exists(nDir):
                        os.mkdir(nDir)
                    classifier.save(saver, nDir)
                    e+=1

            C, D = memory.sampleTe()
            classifier.test(C,D)
            summary_writer.close()





def getdata(memory):

    files=[]
    read(memory,LDIR,np.array([1,0,0]),np.array([0,0,1]))
    read(memory, RDIR,np.array([0,0,1]),np.array([1,0,0]))
    read(memory, SDIR,np.array([0,1,0]),np.array([0,1,0]))
    memory.shaffle()
'''
    imgs, labels = memory.sampleVa()
    print(imgs.shape)
    print(labels.shape)
    print(imgs[0])
    print(labels[0:30])
    imgs, labels = memory.sampleTe()
    print(imgs.shape)
    print(labels.shape)
    print(imgs[0])
    print(labels[0:30])
'''

    #print(img.shape)
    #cv2.imshow('00000001.jpg', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def read(memory,dir,b,breverse):
    print(dir)
    for _, dirs, __ in os.walk(dir):
        print(dirs)
        for name in dirs:
            cdir = os.path.join(dir, name)
            print("begin import from",cdir)
            for _, __, file in os.walk(cdir):
                print(file)  # 当前路径下所有非目录子文件
                files=file
            count = 0
            for i in files:

                CDIR = os.path.join(cdir, i)

                img = cv2.imread(CDIR)
                img3 = img[math.floor(img.shape[0] * 15 / 100):math.floor(img.shape[0] * 85 / 100),
                       math.floor(img.shape[1] * 15 / 100):math.floor(img.shape[1] * 85 / 100)]
                #img = img.astype(float) / 255
                img = cv2.resize(img, (101, 101))
                img3=cv2.resize(img3,(101,101))
                img2 = cv2.flip(img, 1)
                if count < math.floor(len(files) / 40):
                    memory.appendTe(img, b)
                    #memory.appendTe(img2, breverse)
                elif count < math.floor(len(files) / 20):
                    memory.appendVa(img, b)
                    #memory.appendVa(img2, breverse)
                else:
                    memory.append(img, b)
                    memory.append(img2, breverse)
                    memory.append(img3,b)
                count += 1
            memory.printlen(cdir)








if __name__=="__main__":
    main()