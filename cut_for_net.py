import cv2
import os

path = ['./data/train/', './data/val/', './data/test/']
count = 0
for i in path:
    data_list = os.listdir(i)
    for j in data_list:
        count = count + 1
        if count % 200 == 0:
            print('count=' + str(count))
        image = cv2.imread(i + j)
        height_re = image.shape[0] % 8
        width_re = image.shape[1] % 8
        image = image[0:image.shape[0] - height_re, 0:image.shape[1] - width_re]
        cv2.imwrite(i + j, image)
