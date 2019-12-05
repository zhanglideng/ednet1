import torch
import math
import cv2
from utils.loss import *
from utils.ms_ssim import *


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


if __name__ == '__main__':
    # 当前路径下有4张图像，分别测试其SSIM和L1的值
    img1 = torch.from_numpy(cv2.imread('/input/ednet/1.png'))
    img1 = torch.tensor(img1, dtype=torch.float32)
    # print(img1)
    img2 = torch.from_numpy(cv2.imread('/input/ednet/2.png')).to(torch.double)
    img2 = torch.tensor(img2, dtype=torch.float32)
    # print(img2)
    gth1 = torch.from_numpy(cv2.imread('/input/ednet/000000000001.jpg')).to(torch.double)
    gth1 = torch.tensor(gth1, dtype=torch.float32)
    # print(gth1)
    gth2 = torch.from_numpy(cv2.imread('/input/ednet/000000000016.jpg')).to(torch.double)
    gth2 = torch.tensor(gth2, dtype=torch.float32)
    # print(gth2)
    print('图一的SSIM')
    print(ssim(img1 / 255, gth1 / 255))
    print('图二的SSIM')
    print(ssim(img2 / 255, gth2 / 255))
    img3 = torch.stack([img1, img2], dim=0)
    gth3 = torch.stack([gth1, gth2], dim=0)
    print(img3.shape)
    print('图一二堆叠后的SSIM')
    print(ssim(img3 / 255, gth3 / 255))
    print('图一二的SSIM求平均')
    print((ssim(img1 / 255, gth1 / 255) + ssim(img2 / 255, gth2 / 255)) / 2)
    print('图一二的MS-SSIM')
    img1 = torch.transpose(img1, 0, 2)
    gth1 = torch.transpose(gth1, 0, 2)
    img1 = img1.unsqueeze(0).cuda()
    gth1 = gth1.unsqueeze(0).cuda()
    print(img1.shape)
    print(gth1.shape)
    ms_ssim = MS_SSIM(max_val=1)
    print(ms_ssim(img1 / 255, gth1 / 255))
