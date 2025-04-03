import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from piqa import SSIM, LPIPS

class GANLosses:
    def __init__(self, device):
        self.device = device
        self.adv_loss = nn.BCELoss()
        self.vgg = vgg16(pretrained=True).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.pixel_loss = nn.L1Loss()
        self.ssim = SSIM().to(device)
        self.lpips = LPIPS().to(device)
        
    def adversarial_loss(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.adv_loss(pred, target)
    
    def perceptual_loss(self, generated, target):
        gen_rgb = generated.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        gen_rgb = (gen_rgb - mean) / std
        target_rgb = (target_rgb - mean) / std
        gen_features = self.vgg(gen_rgb)
        target_features = self.vgg(target_rgb)
        return F.mse_loss(gen_features, target_features)
    
    def compute_metrics(self, generated, target):
        with torch.no_grad():
            gen = (generated + 1) / 2
            tgt = (target + 1) / 2
            psnr = 10 * torch.log10(1 / F.mse_loss(gen, tgt))
            ssim_val = self.ssim(gen, tgt)
            lpips_val = self.lpips(gen, tgt)
            return {'PSNR': psnr.item(), 'SSIM': ssim_val.item(), 'LPIPS': lpips_val.item()}