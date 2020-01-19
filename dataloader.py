from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio


class EdDataSet(Dataset):
    def __init__(self, transform1, path, batch_size):
        print(path)
        self.transform = transform1
        self.batch_size = batch_size
        # self.transform2 = transform2
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
        # print(image_name)
        image_data = cv2.imread(self.path + '/' + image_name)
        # print(image_data.shape)
        # print(image_name)
        depth_data = np.ones((400, 400, 1), dtype=np.float32) * 255
        a_data = np.ones((400, 400, 1), dtype=np.float32) * 255
        if self.transform:
            input_data = self.transform(image_data)
            gt_data = self.transform(image_data)
            a_data = self.transform(a_data)
            depth_data = self.transform(depth_data)
        else:
            input_data = image_data
            gt_data = image_data
        # item = {'name': image_name, 'input_image': image_data}
        # print(item)
        # print(image_data)
        return input_data, gt_data, a_data, depth_data


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
