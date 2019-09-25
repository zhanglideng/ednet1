import torch


def l2_loss(input_image, output_image):
    return torch.mean(torch.pow((input_image - output_image), 2))*100

