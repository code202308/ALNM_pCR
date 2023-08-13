# import os, random
#
# '''
#     得到图像的标签信息：图片名称 所属family 所属special
# '''
# # path = "/mnt/data3/XiangZhang/code/Unsupervised_object_detection/Dataset/FAIR1M/train_crop_images/Boeing747/"
# # outfile = "/mnt/data3/XiangZhang/code/Unsupervised_object_detection/Classifier/PMG_FG-Aircraft_RS/data/txt/Boeing747.txt"  # 写入的txt文件名
# # 0: A220, 1: A321, 2: A330, 3:A350 , 4:ARJ21, 5:Baseball Field, 6: Basketball Court, 7:Boeing737, 8:, 9:, 10:,
#
# # f = open(outfile, "w")
# # img_name = os.listdir(path)   # 得到图片的名字
# # len_img = len(img_name)  # 计算植物一共的种类
# # for i in range(len_img):
# #     f.write('path/'+'Boeing737/'+img_name[i].split('.')[0] + ' 7'+ "\n")
# # f.close()


import os
import skimage.io
import skimage.color

data_dir = '/mnt/data3/XiangZhang/code/Unsupervised_object_detection/Dataset/FAIR1M/train_crop_images/'  # 文件地址/名称
outfile_train = "/mnt/data3/XiangZhang/code/Unsupervised_object_detection/Classifier/PMG_FG-Aircraft_RS/data/txt/train.txt"  # 写入的txt文件名
outfile_test = "/mnt/data3/XiangZhang/code/Unsupervised_object_detection/Classifier/PMG_FG-Aircraft_RS/data/txt/test.txt"  # 写入的txt文件名
f_w_train = open(outfile_train, "w")
f_w_test = open(outfile_test, "w")
classes = os.listdir(data_dir)
data = []
i = 0
for cls in classes:
    files = os.listdir(data_dir + cls)
    sig = 0
    for f in files:
        if sig%5==0:
            f_w_test.write(str(cls)+'/' + f + ' '+str(i) + "\n")
        else:
            f_w_train.write(str(cls) + '/' + f + ' ' + str(i) + "\n")
        sig = sig + 1
    i = i+1

