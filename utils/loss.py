import torch
from pytorch_ssim._init_ import *


def l2_loss(input_image, output_image):
    return torch.mean(torch.pow((input_image - output_image), 2))


def ssim_loss(input_image, output_image):
    loss = SSIM()
    return 1 - loss(input_image, output_image)
    # return -1 * torch.log(loss(input_image, output_image))
