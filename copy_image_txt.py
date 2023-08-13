# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:57:31 2022
### 专门处理乳腺省医和顺德病例数据
@author: xx
"""
import os
import shutil
import cv2
import numpy as np
import pypinyin

import pandas as pd
# def xinfuzhudata_label(xlsx):
#     m,n = xlsx.shape
#     data = np.zeros((m,n-1))
#     data[:,0] = np.array(xlsx['姓名'])
    
#     data[:,0] = np.array(xlsx['age'])
#     data[:,1] = np.array(xlsx['class'])
#     data[:,2] = np.array(xlsx['ct'])
#     data[:,3] = np.array(xlsx['er'])
#     data[:,4] = np.array(xlsx['pr'])
#     data[:,5] = np.array(xlsx['her1'])
#     data[:,6] = np.array(xlsx['ki67'])
#     label = np.array(xlsx['pcrlabel'])
#     return data,label

# def linbadata_label(xlsx):
#     m,n = xlsx.shape
#     data = np.zeros((m,n-1))
    
#     data[:,0] = np.array(xlsx['姓名'])
#     data[:,1] = np.array(xlsx['年龄'])
#     data[:,2] = np.array(xlsx['病理类型（浸润性导管癌=0，浸润性小叶癌=1，其他类型=2）'])
#     data[:,3] = np.array(xlsx['cT分期(T1=1,T2=2,T3=3,T4=4)'])
#     data[:,4] = np.array(xlsx['ER（阴性=0，阳性=1）'])
#     data[:,5] = np.array(xlsx['PR（阴性=0，阳性=1）'])
#     data[:,6] = np.array(xlsx['HER2（阴性=0，阳性=1，临界=2）'])
#     data[:,7] = np.array(xlsx['Ki-67（<20%=0，≥20%=1）'])
#     label = np.array(xlsx['腋窝淋巴结是否转移（否=0，是=1）'])
#     return data,label

path = 'H:/腋窝淋巴结最新/淋巴结省医病例/'

xls = pd.read_excel(r'H:/腋窝淋巴结转移-数据划分.xlsx',
                        sheet_name='省医病例-训练')

dict1 = xls.loc[:,['姓名','年龄','病理类型（浸润性导管癌=0，浸润性小叶癌=1，其他类型=2）','cT分期(T1=1,T2=2,T3=3,T4=4)','ER（阴性=0，阳性=1）','PR（阴性=0，阳性=1）','HER2（阴性=0，阳性=1）','Ki-67（<20%=0，≥20%=1）','腋窝淋巴结是否转移（否=0，是=1）']]
# clinictxt = open('H:shengyi.txt','a')
arrdict = np.array(dict1)
newclinic = []

path_read = []    #path_read saves all executable files

def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


def check_if_dir(file_path):
    temp_list = os.listdir(file_path)    #put file name from file_path in temp_list

    for temp_list_each in temp_list:

        if os.path.isfile(file_path + '/' + temp_list_each):
            temp_path = file_path + '/' + temp_list_each


            if os.path.splitext(temp_path)[-1] == ('.png'or'.jpg') and os.path.splitext(temp_path)[-1] != '.xml':
                hanzi = temp_path.rsplit('/')[-2]              

                patient_name = pinyin(hanzi)
                save_name = temp_path.rsplit('/')[-1]

                img = cv2.imdecode(np.fromfile((temp_path), dtype=np.uint8), -1)

                cv2.imencode('.jpg', img)[1].tofile('H:/linba_newProcessData_noxml/image/shengyi/'+patient_name[:-4] + ".jpg")  # 英文或中文路径均适用

            if os.path.splitext(temp_path)[-1] == '.xml':    #自己需要处理的是.log文件所以在此加一个判断
            
                hanzi = temp_path.rsplit('/')[-2]
                # print(hanzi)
                patient_name = pinyin(hanzi)
                save_name = temp_path.rsplit('/')[-1]
                                
                idx1 = np.where(arrdict[:,0] == hanzi[0:2])  ### 取xls中汉字前两个和image中的命名对比
                idx2 = np.where(arrdict[:,0] == hanzi[0:3])
                idx3 = np.where(arrdict[:,0] == hanzi[0:4])
                
                if(np.array(idx1).size>0):
                    
                    cln = patient_name[:-4]
                    arrdict[idx1[0],0] = cln
                    clinic = arrdict[idx1[0],:]
                    newclinic.append(clinic[0])
                    # np.savetxt('H:/shengyi.txt',clinic)
                elif(np.array(idx2).size>0):
                    
                    cln = patient_name[:-4]
                    arrdict[idx2[0],0] = cln
                    clinic = arrdict[idx2[0],:]
                    # clinictxt.write(str(clinic[:,1:]-2))
                    # clinictxt.write('\n')
                    # np.savetxt('H:/shengyi.txt',clinic,delimiter=' ',fmt = '%s')  ### 将数据重新拍好顺序再统一保存起来
                    newclinic.append(clinic[0])
                    
                elif(np.array(idx3).size>0):
                    
                    cln = patient_name[:-4]
                    arrdict[idx3[0],0] = cln
                    clinic = arrdict[idx3[0],:]
                    newclinic.append(clinic[0])
                    
                else:
                    print(temp_path.rsplit('/')[-2],'Image名字与excel的名字不符')
                    
                
                
                if(os.path.exists(temp_path.split('.xml')[0] + ".png")):
                    # print("png is exist")
                    img = cv2.imdecode(np.fromfile((temp_path.split('.xml')[0] + ".png"), dtype=np.uint8), -1)
                else:
                    print(temp_path," —— jpg is exist")
                    img = cv2.imdecode(np.fromfile((temp_path.split('.xml')[0] + ".jpg"), dtype=np.uint8), -1)

                cv2.imencode('.jpg', img)[1].tofile('H:/linba_newProcessData/image/shengyi/'+patient_name[:-4] + ".jpg")  # 英文或中文路径均适用
                shutil.copyfile(temp_path, 'H:/linba_newProcessData/xml/shengyi/'+patient_name[:-4] + ".xml")

            else:
                continue
                # save_name = file_path.split('/')[6]
                # img = cv2.imdecode(np.fromfile((temp_path.split('.xml')[0]), dtype=np.uint8), -1)
                # cv2.imencode('.jpg', img)[1].tofile('H:/super1/淋巴/新辅助化疗/classify_data_data1/image_0/' +
                #                                     save_name.encode('unicode_escape').decode('ascii').split('\\')[
                #                                         2] + ".jpg")  # 英文或中文路径均适用

            # path_read.append(temp_path)
        else:
            check_if_dir(file_path + '/' + temp_list_each)    #loop traversal


check_if_dir(path)

# noNan = np.nan_to_num(newclinic)
# noNan = newclinic.copy()
# noNan[np.isnan(noNan)] = 0

np.savetxt('H:/shengyi.txt',newclinic, delimiter=' ',fmt = '%s')  ### 将数据重新拍好顺序再统一保存起来

# f = open('H:/shengyi.txt')
# aa = f.readlines()
# f.close()








