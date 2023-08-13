import os
import pickle
import numpy as np
import PIL.Image
# from PIL import Image
# from tqdm import tqdm
import torch.utils.data


class data_processing(torch.utils.data.Dataset):
    def __init__(self, file_path, train=True, transform=None, target_transform=None):
        self.file_path = file_path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        image_path = os.path.join(self.file_path, '/data/image_data1/')
        if self.train:
            split_f = os.path.join(self.file_path,'/data/txt/train.txt')
        else:
            split_f = os.path.join(self.file_path,'/data/txt/test.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
            # print(file_names)

        self.images = [os.path.join(image_path, x.split(" ")[0]+" "+x.split(" ")[1]+".jpg") for x in file_names]
        self.label = [(int(x.split(" ")[2])) for x in file_names]
        assert (len(self.images) == len(self.label))


    def __getitem__(self, index):

        img = PIL.Image.open(self.images[index]).convert('RGB')
        label = self.label[index]

        if self.transform is not None:
            image = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image,label

    def __len__(self):
        return len(self.images)
