from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import random
from PIL import Image


class EdDataSet(Dataset):
    def __init__(self, transform1, path):
        print(path)
        self.transform = transform1
        self.path = path
        self.data_list = os.listdir(path)
        self.data_list.sort(key=lambda x: int(x[:-4]))
        self.length = len(os.listdir(self.path))

    def __len__(self):
        return self.length

    @staticmethod
    def random_flip(image):
        """
        new_im = transforms.RandomHorizontalFlip(p=0.5)(im)  # p表示概率 水平翻转

        # 90度，180度，270度旋转

        transforms.RandomApply(transforms, p=0.5)
        功能：给一个transform加上概率，以一定的概率执行该操作

        8.随机旋转：transforms.RandomRotation
        class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
        功能：依degrees随机旋转一定角度
        参数：
        degress- (sequence or float or int) ，若为单个数，如 30，则表示在（-30，+30）之间随机旋转
        若为sequence，如(30，60)，则表示在30-60度之间随机旋转
        """
        rotate = random.randint(0, 3)
        image = transforms.RandomHorizontalFlip()(image)
        image = transforms.RandomRotation(rotate * 90)(image)
        return image

    def __getitem__(self, idx):
        """
            need dehazy image
        """
        image_name = self.data_list[idx]
        # print(image_name)
        image_data = Image.open(self.path + '/' + image_name)
        # print(image_data.shape)
        # print(image_name)
        t_data = np.ones((400, 400, 1), dtype=np.float32) * 255
        # 测一下这里会被归一化吗
        a_data = np.ones((400, 400, 3), dtype=np.float32) * 255
        if self.transform:
            input_data = self.transform(image_data)
            gt_data = self.transform(image_data)
            a_data = self.transform(a_data)
            t_data = self.transform(t_data)
        else:
            input_data = image_data
            gt_data = image_data
        print(t_data)
        print(a_data)
        return input_data.cuda(), gt_data.cuda(), a_data.cuda(), t_data.cuda()


if __name__ == '__main__':
    train_path = './data/train/'
    validation_path = './data/val/'
    test_path = './data/test/'
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    a1 = np.array([[[255, 255, 255],
                    [255, 255, 255]],
                   [[255, 255, 255],
                    [255, 255, 255]]], dtype=float)
    b1 = cv2.imread('./test.png')
    print(a1)
    print(b1)
    # data = EdDataSet(transform, train_path)
    # data_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    count = 0
    a2 = transform(a1)
    b2 = transform(b1)
    print(a2)
    print(b2)
    # for i in data_loader:
    #    image = i
    #    print('image.shape:' + str(image.shape))
    #    count += 1
    # print('count:' + str(count))
