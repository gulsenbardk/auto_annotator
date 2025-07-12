from header import *
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, num_classes):
     
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).float()  # shape: (B*N, C)
        probs = torch.softmax(logits, dim=1)                       # shape: (B*N, C)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # shape: (B*N,)
        p_t = (probs * targets_one_hot).sum(dim=1)                    # shape: (B*N,)
        focal_factor = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, num_classes):
        probs = F.softmax(logits, dim=-1)  # (B, N, C)
        one_hot = F.one_hot(targets, num_classes).float()  # (B, N, C)

        intersection = (probs * one_hot).sum(dim=(0, 1))
        union = probs.sum(dim=(0, 1)) + one_hot.sum(dim=(0, 1))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
