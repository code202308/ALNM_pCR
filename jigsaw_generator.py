# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 19:23:09 2022

@author: Administrator
"""

import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
import SimpleITK as sitk

def resize_image_itk(itkimage, newSize, interpolator):
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(_SITK_INTERPOLATOR_DICT[interpolator])
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                l.append([a, b, c])
    xblock_size = 512 // n
    yblock_size = 512 // n
    zblock_size = 256 // n
    
    rounds = n ** 3
    random.shuffle(l)
    jigsaws = images.copy()
    for i in range(rounds):
        z, y, x = l[i]
        temp = jigsaws[..., 0:zblock_size, 0:yblock_size, 0:xblock_size].copy()
        jigsaws[..., 0:zblock_size, 0:yblock_size, 0:xblock_size] = jigsaws[..., z * zblock_size:(z + 1) * zblock_size,
                                                y * yblock_size:(y + 1) * yblock_size, x * xblock_size:(x + 1) * xblock_size].copy()
        jigsaws[..., z * zblock_size:(z + 1) * zblock_size, y * yblock_size:(y + 1) * yblock_size, x * xblock_size:(x + 1) * xblock_size] = temp

    return jigsaws

path = 'C:/Users/Administrator/Desktop/coro.nii.gz'
itk_coro =sitk.ReadImage(path)
img = resize_image_itk(itk_coro,(512,512,256),'linear')
img_coro = sitk.GetArrayFromImage(img)

out = jigsaw_generator(img_coro, 2)



img = cv.imread('C:/Users/Administrator/Desktop/1.jpg')
img1 = cv.resize(img,(448,448),interpolation=cv.INTER_CUBIC)

img2 = np.transpose(img1,(2,1,0))
inputs1 = jigsaw_generator(img2, 4)
# plt.imshow(img1)
plt.imshow(np.transpose(inputs1,(2,1,0)))


x = np.zeros((6,3,64))
x[0,:,:] = 0
x[1,:,:] = 1
x[2,:,:] = 2
x[3,:,:] = 3
x[4,:,:] = 4
x[5,:,:] = 5

xy = [i for i in range(6)]
y = [i for i in range(6)]
random.shuffle(y)

z = x[y,:,:]

a= z.copy()
b = np.argsort(y)
a1 = a[b,:,:]

