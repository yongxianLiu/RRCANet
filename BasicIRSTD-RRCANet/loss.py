import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge

class MultiBCEloss(nn.Module):
    def __init__(self):
        super(MultiBCEloss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            bce_loss = nn.BCELoss(size_average=True)
            for i in range(len(preds)):
                pred = preds[i]

                loss_total = loss_total + bce_loss(pred, gt_masks)
            return loss_total / len(preds)

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
    def forward(self, preds, gt_masks):
        smooth = 1.0

        i_flat = preds.view(-1)
        t_flat = gt_masks.view(-1)

        intersection = (i_flat * t_flat).sum()

        loss = 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
        return loss

class TopKLoss(nn.Module):
    def __init__(self):
        super(TopKLoss, self).__init__()
    def forward(self, preds, gt_masks):
        k = 10
        res = nn.BCELoss(reduction='none')(preds, gt_masks)
        num_pixels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1,)), int(num_pixels * k / 100), sorted=False)
        return res.mean()

class DiceTopk(nn.Module):
    def __init__(self):
        super(DiceTopk, self).__init__()
        self.dice = SoftDiceLoss()
        self.TopK = TopKLoss()

    def forward(self, preds, gt_masks):
        dice_loss = self.dice(preds, gt_masks)
        topk_loss = self.TopK(preds, gt_masks)
        dicetopk = dice_loss + topk_loss
        return dicetopk


class PolyTopkLoss(nn.Module):
    def __init__(self):
        super(PolyTopkLoss, self).__init__()
    def forward(self, preds, gt_masks):
        k = 10
        epsilon = 3.1  ##>=-1
        BCE_loss = nn.BCELoss(reduction='none')(preds, gt_masks)
        pt = torch.exp(-BCE_loss)
        poly1 = BCE_loss + (1 - pt) * epsilon

        num_pixels = np.prod(poly1.shape)
        loss, _ = torch.topk(poly1.view((-1,)), int(num_pixels * k / 100), sorted=False)

        return loss.mean()



class DicepolyTopk(nn.Module):
    def __init__(self):
        super(DicepolyTopk, self).__init__()
        self.dice = SoftDiceLoss()
        self.polyTopk = PolyTopkLoss()

    def forward(self, preds, gt_masks):
        dice_loss = self.dice(preds, gt_masks)
        polytopk = self.polyTopk(preds, gt_masks)
        return dice_loss + polytopk