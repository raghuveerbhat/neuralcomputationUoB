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
    def __init__(self, alpha=0.9, gamma=2):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets) / (torch.sum(preds) + torch.sum(targets))
        return dice_loss.mean()