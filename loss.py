import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torchvision.models as models
import torchvision.transforms as transforms


# -----------------------------
# Charbonnier Loss
# -----------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon)
        return loss.mean()


# -----------------------------
# SSIM Loss
# -----------------------------
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # SSIM returns similarity, so loss = 1 - SSIM
        return 1 - ssim(pred, target)

# -----------------------------
# Combined Denoising Loss
# -----------------------------
class DenoisingLoss(nn.Module):
    def __init__(self, w_charb=1, w_ssim=0):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.w_charb = w_charb
        self.w_ssim = w_ssim

    def forward(self, pred, target):
        loss_charb = self.charb(pred, target)
        loss_ssim = self.ssim(pred, target)

        loss = (
            self.w_charb * loss_charb +
            self.w_ssim * loss_ssim
        )

        # For logging individual components
        self.charb_ind = loss_charb.item()
        self.ssim_ind = loss_ssim.item()

        return loss

