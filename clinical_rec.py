# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:39:14 2022

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def numeric_score(gt,pred):
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    tpr = TP/(TP+FN)
    tnr = TN/(TN+FP) 
    npv = TN/(TN+FN) 
    ppv = TP/(TP+FP)
    acc = (TP+TN)/(TP+FP+FN+TN)
    recall = tpr
    pr = TP / (TP + FP + 1e-10)
    f1 = 2*pr*recall/(pr+recall)
  
    return tpr,tnr,npv,ppv,acc

def xinfuzhudata_label(xlsx):
    m,n = xlsx.shape
    data = np.zeros((m,n-2))
    data[:,0] = np.array(xlsx['age'])
    data[:,1] = np.array(xlsx['class'])
    data[:,2] = np.array(xlsx['ct'])
    data[:,3] = np.array(xlsx['er'])
    data[:,4] = np.array(xlsx['pr'])
    data[:,5] = np.array(xlsx['her1'])
    data[:,6] = np.array(xlsx['ki67'])
    label = np.array(xlsx['pcrlabel'])
    return data,label


def linbadata_label(xlsx):
    m,n = xlsx.shape
    data = np.zeros((m,n-2))
    data[:,0] = np.array(xlsx['age'])
    data[:,1] = np.array(xlsx['class'])
    data[:,2] = np.array(xlsx['cT'])
    data[:,3] = np.array(xlsx['er'])
    data[:,4] = np.array(xlsx['pr'])
    data[:,5] = np.array(xlsx['her2'])
    data[:,6] = np.array(xlsx['ki67'])
    label = np.array(xlsx['linba_label'])
    return data,label

xls = pd.read_excel('C:/Users/Administrator/Desktop/linba/shengyi_train.xlsx')
xls_t = pd.read_excel('C:/Users/Administrator/Desktop/linba/shunde_test.xlsx')

max_min_scaler = lambda x:(x-np.min(x))/(np.max(x)-np.min(x))
xls['age'] = xls[['age']].apply(max_min_scaler)
xls_t['age'] = xls_t[['age']].apply(max_min_scaler)

traindata,trainlabel = xinfuzhudata_label(xls)
testdata,testlabel = xinfuzhudata_label(xls_t)

traindata,trainlabel = linbadata_label(xls)
testdata,testlabel = linbadata_label(xls_t)

###################################################3
model=LogisticRegression(solver='liblinear',multi_class='ovr')    
model.fit(traindata,trainlabel) 
y_pre=model.predict_proba(testdata)
pro_label = np.argmax(y_pre, axis = 1)
pro_b = np.max(y_pre, axis = 1)
y_0=list(y_pre[:,1])   
fpr,tprr,thresholds=roc_curve(testlabel,y_0)  
tpr,tnr,npv,ppv,acc = numeric_score(testlabel,pro_label)
print('----------------------------')
print(tpr,tnr,npv,ppv,acc)
print('----------------------------')
auc=roc_auc_score(testlabel,y_0) 
print(auc)


plt.rcParams['font.sans-serif'] = ['Arial']  
plt.figure(figsize=(10,8))
plt.title('YN Train - ROC Curve',fontsize=14)
plt.plot(fpr,tprr,'b',linewidth=1.5)
# plt.plot(fpr,tpr,'b',label = 'AUC = %0.3f' % auc,linewidth=1.5)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xticks(fontsize=13)  # 默认字体大小为10
plt.yticks(fontsize=13)

plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.show()

