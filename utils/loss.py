import torch
import utils.ssim


def l2_loss(input_image, output_image):
    return torch.mean(torch.pow((input_image - output_image), 2))


def ssim_loss(input_image, output_image):
    # losser = utils.ssim.SSIM(data_range=1., channel=3)
    # loss = losser(input_image, output_image).mean()
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
