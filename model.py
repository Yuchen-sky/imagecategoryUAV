import numpy as np
import tensorflow as tf
import os


class Classifier:
    def __init__(self,sess,state_shape,lr=1e-3,seed=10):
        #self.batch_size=512
        self.sess=sess
        self.initlr=lr
        #self.lr=lr
        tf.set_random_seed(seed)
        self.state_shape=state_shape
        self.total_loss=0
        self.val_acc=0
        self.global_ = tf.Variable(tf.constant(0), trainable=False)
        with tf.name_scope("learning_rate"):
            self.lr=tf.train.exponential_decay(
                self.initlr,
                self.global_,
                decay_steps=300,
                decay_rate=0.95,
                staircase=True,
                name="lr"
            )
        with tf.name_scope('Input'):
            self.img=tf.placeholder(tf.float32,[None,101,101,3],name="X_placeholder")
            self.label=tf.placeholder(tf.int32,[None,3],name="Y_placeholder")
        with tf.variable_scope("classifier"):
            self.eval_net=self.build_network(self.img,"realclassifier")
        self.eval = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="classifier/realclassifier")
        with tf.name_scope('Loss'):
            self.entropy=tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.eval_net[0],name="cross_entropy")
            self.loss=tf.reduce_mean(self.entropy,name="loss")
        with tf.name_scope('Optimization'):
            self.train_step=tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.name_scope('Evaluate'):
            self.correct_prediction=tf.equal(tf.argmax(self.eval_net[1],1),tf.argmax(self.label,1))
            self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        with tf.name_scope('Predict'):
            self.getinfo=self.eval_net[1]

        #self.sess.run(tf.global_variables_initializer())






    def build_network(self,img,scope):

        with tf.variable_scope(scope):
            init_w1 =tf.random_uniform_initializer(-0.05, 0.05)# tf.truncated_normal_initializer(0., 3e-4)
            init_w2 = tf.random_uniform_initializer(-0.05, 0.05)

            conv1 = tf.layers.conv2d(img,32,[4,4],[1,1],kernel_initializer=init_w1,activation=tf.nn.tanh)
            pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
            conv2 = tf.layers.conv2d(pool1,32,[4,4],[1,1] ,kernel_initializer=init_w1,activation=tf.nn.tanh)
            pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
            conv3 = tf.layers.conv2d(pool2,32,[4,4],[1,1] ,kernel_initializer=init_w1,activation=tf.nn.tanh)
            pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=2)
            conv4 = tf.layers.conv2d(pool3,32,[4,4],[1,1] ,kernel_initializer=init_w1,activation=tf.nn.tanh)
            pool4 = tf.layers.max_pooling2d(inputs=conv4,pool_size=[2,2],strides=2)
            flatten = tf.layers.flatten(pool4)

            fl = tf.layers.dense(inputs=flatten,units=200,activation=tf.nn.tanh,kernel_initializer=init_w2,name="FullLayer1")
            fl2 = tf.layers.dense(inputs=fl,units=3,kernel_initializer=init_w2,name="FullLayer2")
            fl2_soft=tf.nn.softmax(fl2)
        return fl2,fl2_soft




    def caculate(self,X,Y,i):
        #print("----------------------------begin caculate--------------------")
        _,loss_batch=self.sess.run([self.train_step,self.loss],feed_dict={self.img:X,self.label:Y})
        t=self.sess.run(self.lr,feed_dict={self.global_:i})
        if i%50==0:
            print('step: {}, train_loss: {},learning rate:{}'.format(i, loss_batch,t).ljust(20, " "))
            #self.lr=self.lr*0.95
        self.total_loss+=loss_batch



    def statistic(self,X,Y,i,Z,B,C):
        #print("----------------------------statistic--------------------")
        self.val_acc,z=self.sess.run([self.accuracy,Z],feed_dict={self.img:X,self.label:Y})
        trainacc = self.sess.run(self.accuracy, feed_dict={self.img: B, self.label: C})
        print('step: {}, total_average_train_loss: {}, val_acc: {}, train_acc: {}'.format(i,self.total_loss/200,self.val_acc,trainacc).ljust(20, " "))
        self.total_loss=0
        return z




    def test(self,X,Y):
        print("----------------------------test--------------------")
        test=self.sess.run(self.accuracy,feed_dict={self.img:X,self.label:Y})
        print('test accuracy:{}'.format(test).ljust(20," "))


    def predict(self,X):
        _,predictit=self.sess.run(self.eval_net, feed_dict={self.img: X})
        return predictit



    def save(self,saver, dir):
        path = os.path.join(dir, 'model')
        saver.save(self.sess,path)


    def load(self,saver,dir):
        e=str(len([file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir,file))])-1)
        print("load model in file: ",e)
        path=os.path.join(dir,e)
        path=os.path.join(path,'checkpoint')
        print(path)
        ckpt=tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess,ckpt.model_checkpoint_path)
            print("success")
            return True
        return False