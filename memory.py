import pickle
import os
import random
from collections import deque
import numpy as np



class Memory:
    def __init__(self,memory_size=200000):
        self.buffer=list()
        self.validation=list()
        self.test=list()
        self.memory_size=memory_size

    def append(self,img,label):
        if len(self.buffer)<self.memory_size:
            self.buffer.append((img,label))
        else:
            print("buffer is enough")
        print(len(self.buffer))


    def sample(self,size):
        minibatch=random.sample(self.buffer,size)
        imgs = np.array([data[0] for data in minibatch])
        labels = np.array([data[1] for data in minibatch])
        return imgs,labels

    def shaffle(self):
        random.shuffle(self.buffer)
        random.shuffle(self.validation)
        random.shuffle(self.test)

    def appendVa(self,img,label):
        self.validation.append([img,label])
        print(len(self.validation)," validation set has got")



    def sampleVa(self):
        #print(len(self.validation), "get validation set ")
        imgs = np.array([data[0] for data in self.validation])
        labels = np.array([data[1] for data in self.validation])
        return imgs,labels

    def appendTe(self, img, label):
        self.test.append([img, label])
        print(len(self.test), " test set has got")

    def sampleTe(self):
        print(len(self.test), "get test set ")
        imgs = np.array([data[0] for data in self.test])
        labels = np.array([data[1] for data in self.test])
        return imgs, labels

    def printlen(self,path):
        print(path)
        print("buffer",len(self.buffer))
        print("validation",len(self.validation))
        print("test",len(self.test))

