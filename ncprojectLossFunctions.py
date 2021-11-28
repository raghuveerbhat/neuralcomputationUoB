import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CEloss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        ce_loss = self.CEloss(preds, targets) 
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        return 1 - dice_loss.mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CEloss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        # focal loss
        ce_loss = self.CEloss(preds, targets) 
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

        # dice loss
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        dice_loss = 1 - dice_loss.mean()
       
        loss = focal_loss + dice_loss
        return loss