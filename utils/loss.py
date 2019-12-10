import torch
from utils.ms_ssim import *


def l2_loss(input_image, output_image):
    l1_loss_fn = torch.nn.L1Loss(reduction='mean')
    return l1_loss_fn(input_image, output_image)


def ssim_loss(input_image, output_image):
    losser = MS_SSIM(max_val=1)
    # losser = MS_SSIM(data_range=1.).cuda()
    return 1 - losser(input_image, output_image)
