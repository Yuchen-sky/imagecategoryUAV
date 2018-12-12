# imagecategoryUAV
imagecategoryv2
this is a implementation of paper :(main configure detail can be find in this paper)
A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots https://ieeexplore.ieee.org/document/7358076

you can run main.py to train your network and see specific state in tensorblord. 


for deviding video into images, this function can be find in functiontest.py

firstly, download dataset from http://bit.ly/perceivingtrails
then put all the lc image reporitories in 000/video/lc/left.frame, 
then also take rc sc images in same way. after running main.py it can train and save model in data/savemodel,
the validation dataset and test dataset will be set automatically.

you also can use realtest.py to show what this model used in other dataset.

predict output
![Image text](https://github.com/Yuchen-sky/imagecategoryUAV/blob/master/realtest_output/1.png?raw=true)
![Image text](https://github.com/Yuchen-sky/imagecategoryUAV/blob/master/realtest_output/4.png?raw=true)
