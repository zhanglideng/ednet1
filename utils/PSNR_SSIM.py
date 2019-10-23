import torch
import math


def ssim(input_image, output_image):
    mean_ii = torch.mean(input_image)
    mean_oi = torch.mean(output_image)
    var_ii = torch.var(input_image)
    var_io = torch.var(output_image)
    mean_ioi = torch.mean(input_image * output_image)
    c1 = 0.0001
    c2 = 0.0009
    var = mean_ioi - mean_ii * mean_oi
    ssim = (2 * mean_ii * mean_oi + c1) * (2 * var + c2) / (mean_ii * mean_ii + mean_oi * mean_oi + c1) / (
            var_ii + var_io + c2)
    return ssim


def psnr(input_image, output_image):
    mse = torch.mean(torch.pow((input_image - output_image), 2))
    return 10 * torch.log(255 / mse) / math.log(10)
