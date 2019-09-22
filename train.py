# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import EdDataSet
from ED_model import CNN
import time
import xlwt

LR = 0.001
EPOCH = 300
BATCH_SIZE = 4
excel_line = 1
train_path = './data/train/'
validation_path = './data/validation/'
save_path = './checkpoints/best_cnn_model.pt'
excel_save = './result.xls'

# 初始化excel
f, sheet = init_excel()
# 加载模型
net = CNN()
net = net.cuda()
print(net)

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
# 读取训练集数据
train_data = EdDataSet(transform, train_path)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 读取验证集数据
validation_data = EdDataSet(transform, validation_path)
validation_data_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

saving_index = 0
flag = False
min_loss = 999999999
min_epoch = 0
no_update = 0  # 有多少轮未更新最佳网络
LR_flag = 0
temple_loss = 0
start_time = time.time()
# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    train_epo_loss = 0
    validation_epo_loss = 0
    for input_image in train_data_loader:
        saving_index += 1
        index += 1
        input_image = input_image.cuda()
        optimizer.zero_grad()
        output_image = net(input_image)
        loss = l2_loss(output_image, input_image)
        loss.backward()
        iter_loss = loss.item()
        train_epo_loss += iter_loss
        optimizer.step()
        # 在这里建一个progressbar
        if np.mod(index, 100) == 0:
            print('epoch %d, %03d/%d, loss=%.5f' % (
                epoch + 1, index, len(train_data_loader), iter_loss))
            print_time(start_time, index, EPOCH, len(train_data_loader), (train_epo_loss - temple_loss) / 100)
            temple_loss = train_epo_loss

    # 验证集的损失计算
    with torch.no_grad():
        for input_image in validation_data_loader:
            input_image = input_image.cuda()
            output_image = net(input_image)
            loss = l2_loss(output_image, input_image)
            iter_loss = loss.item()
            validation_epo_loss += iter_loss
    train_epo_loss = train_epo_loss / len(train_data_loader)
    validation_epo_loss = validation_epo_loss / len(validation_data_loader)
    print('\nepoch %d train loss = %.5f' % (epoch + 1, train_epo_loss))
    print('epoch %d validation loss = %.5f' % (epoch + 1, validation_epo_loss))
    excel_line = write_excel(sheet=sheet,
                             excel_line=excel_line,
                             epoch_to_save=epoch,
                             train_loss_to_save=train_epo_loss,
                             validation_loss_to_save=validation_epo_loss,
                             learning_rate=LR)
    f.save(excel_save)

    if validation_epo_loss < min_loss:
        min_loss = validation_epo_loss
        min_epoch = epoch
        torch.save(net, save_path)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
        no_update = 0
        LR_flag = 0
    else:
        print('not improve for epoch %d with %.5f' % (min_epoch, min_loss))
        no_update += 1
        if no_update >= 50:
            break
        if no_update - LR_flag >= 5:
            LR = LR * 0.9
            LR_flag = no_update
    print('learning rate is ' + str(LR) + '\n')
print('Train is Done!')
