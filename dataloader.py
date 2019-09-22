from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio


class EdDataSet(Dataset):
    def __init__(self, transforms, path):
        print(path)
        self.transforms = transforms
        # 读取无雾图
        self.path = path
        self.data_list = os.listdir(path)
        self.data_list.sort(key=lambda x: int(x[:-4]))

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        """
            need dehazy image
        """
        image_name = self.data_list[idx]
        image_data = cv2.imread(self.path + '/' + image_name)
        if self.transforms:
            image_data = self.transforms(image_data)
        return image_data


if __name__ == '__main__':
    train_path = './data/train/'
    validation_path = './data/validation/'
    test_path = './data/test/'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    data = EdDataSet(transform, train_path)
    data_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    count = 0
    for i in data_loader:
        image = i
        print('image.shape:' + str(image.shape))
        count += 1
    print('count:' + str(count))
