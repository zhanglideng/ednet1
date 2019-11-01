import cv2
import os
import numpy as np
import math

count = 0
path = ['./data/train/']
image_list = os.listdir(path)
for i in range(len(image_list)):
    image = cv2.imread(path + image_list[i])
    print(image.shape)
    if i % 500 == 0:
        print(i)
    if image.shape[0] >= 400 and image.shape[1] >= 400:
        count += 1
print(count)
