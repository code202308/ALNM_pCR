import os
import shutil
import cv2
import numpy as np
path = 'H:/super1/淋巴/新辅助化疗/新辅助云南病例'
path_read = []    #path_read saves all executable files

def check_if_dir(file_path):
    temp_list = os.listdir(file_path)    #put file name from file_path in temp_list
    
    
    for temp_list_each in temp_list:

        if os.path.isfile(file_path + '/' + temp_list_each):
            temp_path = file_path + '/' + temp_list_each
            if os.path.splitext(temp_path)[-1] == '.xml':    #自己需要处理的是.log文件所以在此加一个判断
                save_name = file_path.split('/')[6]
                img = cv2.imdecode(np.fromfile((temp_path.split('.xml')[0] + ".png"), dtype=np.uint8), -1)

                cv2.imencode('.jpg', img)[1].tofile('H:/super1/淋巴/新辅助化疗/data_img_xml/img/' + save_name.encode('unicode_escape').decode('ascii').split('\\')[2] + ".jpg")  # 英文或中文路径均适用
                shutil.copyfile(temp_path, 'H:/super1/淋巴/新辅助化疗/data_img_xml/xml/'+save_name.encode('unicode_escape').decode('ascii').split('\\')[2] + ".xml")
            else:
                continue
            # path_read.append(temp_path)
        else:
            check_if_dir(file_path + '/' + temp_list_each)    #loop traversal


check_if_dir(path)
