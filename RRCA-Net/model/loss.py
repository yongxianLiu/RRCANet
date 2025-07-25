import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

#IoU
def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1.0

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        loss = 1 - loss.mean()
        return loss


#Dice
def Softdice_loss(prediction, target):
    smooth = 1.0
    prediction = torch.sigmoid(prediction)

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()
    loss = 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
    return loss



#topk
def TopKLoss(pred, target):
    k = 10
    res = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    num_pixels = np.prod(res.shape)
    res, _ = torch.topk(res.view((-1,)), int(num_pixels * k / 100), sorted=False)
    return res.mean()

#dicetopk
def DiceTopK(pred, target):
    dice_loss = Softdice_loss(pred, target)
    topk_loss = TopKLoss(pred, target)
    dicetopk = dice_loss + topk_loss
    return  dicetopk


#polyTopk_ce
def PolyTopk(inp, target):
    k = 10
    epsilon = 3.1 ##>=-1
    inp = torch.nn.Sigmoid()(inp)
    BCE_loss = nn.BCELoss(reduction='none')(inp, target)
    pt = torch.exp(-BCE_loss)
    poly1 = BCE_loss+(1-pt)*epsilon

    num_pixels = np.prod(poly1.shape)
    loss, _ = torch.topk(poly1.view((-1,)), int(num_pixels * k / 100), sorted=False)
    return loss.mean()


#DicepolyTopK
def DicepolyTopK(inp, target):
    dice_loss = Softdice_loss(inp, target)
    polytopk = PolyTopk(inp, target)
    return dice_loss+polytopk





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


