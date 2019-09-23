import sys
import argparse
import time
import glob
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from torchvision import transforms
from dataloader import EdDataSet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from ED_model import CNN
import torch

test_path = './data/train/'
BATCH_SIZE = 1


def l2_loss(im_input, im_output):
    return torch.mean(torch.pow((im_input - im_output), 2)) * 100


def get_image_for_save(img):
    img = img.numpy()
    img = np.squeeze(img)
    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img


save_path = 'result_{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_path = './checkpoints/best_cnn_model.pt'
net = torch.load(model_path)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
test_data = EdDataSet(transform, test_path)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(">>Start testing...\n")

avg_elapsed_time = 0.0
count = 0
for input_image in test_data_loader:
    count += 1
    print(">>Processing ./{}".format(str(count)))
    with torch.no_grad():
        net = net.cuda()
        # net.train(False)
        # net.eval()
        # 上面这两句话到底有什么问题
        input_image = input_image.cuda()
        start_time = time.time()
        output_image = net(input_image)
        print(l2_loss(output_image, input_image).item())
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time
    output_image = output_image.cpu()
    im_output_for_save = get_image_for_save(output_image)
    filename = str(count) + '.png'
    cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

print(">>Finished!"
      "It takes average {}s for processing single image\n"
      "Results are saved at ./{}".format(avg_elapsed_time / count, save_path))
