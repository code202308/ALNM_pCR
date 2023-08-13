from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import data_processing
from utils import *

def test(model_path=None):
    # Model
    net = torch.load(model_path)

    # GPU
    device = torch.device("cuda:1")
    # device = torch.device("cuda:0,1")
    net.to(device)
    # cudnn.benchmark = True

    val_acc, val_acc_com, val_loss = val(net,8)
    with open('/model_data1' + '/results_val.txt', 'a') as file:
        file.write('final_model, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (val_acc, val_acc_com, val_loss))


test(model_path='/model_data1/model.pth')         # the saved model where you want to resume the training


