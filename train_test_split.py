# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:32:54 2022

@author: ZhangXiao
"""

import os
import shutil
import cv2
import numpy as np
import pypinyin
import pandas as pd
from sklearn.model_selection import train_test_split
path = 'H:/data_20220620/linba_newProcessData/'

image = path + 'image/shengyi/' 
xml = path + 'xml/shengyi/' 
txt = path+ 'linba_shengyi.txt'

f = open(txt)
txtlist = f.readlines()
f.close()

x_train,x_test,y_train,y_test=train_test_split(txtlist,txtlist,test_size=0.x)



train = [x.split(' ') for x in x_train]
n = len(train)
for i in range(len(train)):
    if (len(train[i])>9):  ## >9说明原始的名字和
        print()
    
    name = train[i][0]
    
    
    pathImg = image+name+'.jpg'
    pathxml = xml+name+'.xml'  
    shutil.copyfile(pathImg, path+'train/'+name+ ".jpg")
    shutil.copyfile(pathxml, path+'train_label/'+name+ ".xml")
    

test = [x.split(' ') for x in x_test]
for i in range(len(test)):
    
    name1 = test[i][0]
    pathImg1 = image+name1+'.jpg'
    pathxml1 = xml+name1+'.xml'  
    shutil.copyfile(pathImg1, path+'test/'+name1+ ".jpg")
    shutil.copyfile(pathxml1, path+'test_label/'+name1+ ".xml")


savetrain = [x[:-1] for x in x_train]
savetest = [x[:-1] for x in x_test]

np.savetxt(path+'train.txt',savetrain, delimiter=' ',fmt = '%s')  ### 将数据重新拍好顺序再统一保存起来
np.savetxt(path+'test.txt',savetest, delimiter=' ',fmt = '%s')  ### 将数据重新拍好顺序再统一保存起来

