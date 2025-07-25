from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        # elif model_name == 'DNANet_BY':
            # if mode == 'train':
            #     self.model = DNAnet_BY(mode='train')
            # else:
            #     self.model = DNAnet_BY(mode='test')
        elif model_name == 'ACM':
            if mode == 'train':
                self.model = ACM(mode='train')
            else:
                self.model = ACM(mode='test')
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        # elif model_name == 'ISNet':
        #     if mode == 'train':
        #         self.model = ISNet(mode='train')
        #     else:
        #         self.model = ISNet(mode='test')
        #     self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
            self.cal_loss = MultiBCEloss()
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ResUNet':
            self.model = ResUNet()
        elif model_name == 'ResUNet_RuCB':
            self.model = ResUNet_RuCB()
        elif model_name == 'RRCANet':
            self.model = RRCANet()
            self.cal_loss = DicepolyTopk()
        elif model_name == 'AMFU':
            if mode == 'train':
                self.model = AMFU(mode='train')
            else:
                self.model = AMFU(mode='test')
        # elif model_name == 'AMFU_RuCB':
        #     if mode == 'train':
        #         self.model = AMFU_RuCB(mode='train')
        #     else:
        #         self.model = AMFU_RuCB(mode='test')
        # elif model_name == 'DNANet_RuCB':
        #     self.model = DNANet_RuCB()
        elif model_name == 'LW-IRSTNet':
            self.model = LW_IRSTNet()



    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
