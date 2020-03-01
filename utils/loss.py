import torch
from utils.ms_ssim import *


def l2_loss(input_image, output_image):
    l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
    return l2_loss_fn(input_image, output_image) * 100


def ssim_loss(input_image, output_image):
    losser = MS_SSIM(max_val=1).cuda()
    # losser = MS_SSIM(data_range=1.).cuda()
    return (1 - losser(input_image, output_image)) * 100


def loss_function(image, weight):
    output, gt = image
    loss_train = [l2_loss(gt, output),
                  ssim_loss(gt, output)]
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
        loss_train[i] = loss_train[i].item()
    return loss_sum, loss_train
