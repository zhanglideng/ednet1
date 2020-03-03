# -*- coding: utf-8 -*-
import sys

sys.path.append('/home/aistudio/external-libraries')

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import *
from Res_ED_model import CNN
import time
import xlwt
from utils.ms_ssim import *
import os

LR = 0.001  # 学习率
EPOCH = 40  # 轮次
BATCH_SIZE = 16  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
weight = [1, 1]  # 损失函数的权重
loss_num = 2  # 损失函数的数量
accumulation_steps = 1  # 梯度积累的次数，类似于batch-size=64
itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 1024 // BATCH_SIZE  # 训练64次后保存相关数据到excel
dataset_size = ''
train_path = '/home/aistudio/data/data20016/cut_coco/' + dataset_size + 'train/'  # 训练集的路径
val_path = '/home/aistudio/data/data20016/cut_coco/' + dataset_size + 'val/'  # 验证集的路径
save_path = './result_ednet_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
excel_save = save_path + 'result.xls'  # 保存excel的路径
save_model = save_path + 'model.pt'

# 初始化excel
f, sheet_train, sheet_val = init_excel()
net = CNN(64)
net = net.cuda()

if not os.path.exists(save_path):
    os.makedirs(save_path)
# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])
# 读取训练集数据
train_data = EdDataSet(transform, train_path)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 读取验证集数据
val_data = EdDataSet(transform, val_path)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

min_loss = 999999999
min_epoch = 0
itr = 0
start_time = time.time()
# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    loss = 0
    train_loss = 0
    val_loss = 0
    loss_excel = [0] * loss_num
    net.train()
    print('start train!')
    for input_image, gt_image, a_image, gt_depth in train_data_loader:
        index += 1
        itr += 1
        output_image, scene_feature = net(input_image, a_image, gt_depth)
        loss_image = [output_image, gt_image]
        loss, temp_loss = loss_function(loss_image, weight)
        train_loss += loss.item()
        print(temp_loss)
        for i in range(len(temp_loss)):
            loss_excel[i] = loss_excel[i] + temp_loss[i]
        loss = loss / accumulation_steps
        loss.backward()
        if ((index + 1) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:
            # print(index)
            # print(itr_to_excel)
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d, loss=%.5f' % (epoch + 1, index, len(train_data_loader), sum(loss_excel)))
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel(sheet=sheet_train,
                                           data_type='train',
                                           line=excel_train_line,
                                           epoch=epoch,
                                           itr=itr,
                                           loss=loss_excel,
                                           weight=weight)
            f.save(excel_save)
            loss_excel = [0] * loss_num
    optimizer.step()
    optimizer.zero_grad()
    # 验证集的损失计算
    loss_excel = [0] * loss_num
    with torch.no_grad():
        net.eval()
        print('\nstart val!')
        for input_image, gt_image, a_image, gt_depth in val_data_loader:
            output_image, scene_feature = net(input_image, a_image, gt_depth)
            loss_image = [output_image, gt_image]
            loss, temp_loss = loss_function(loss_image, weight)
            val_loss += loss.item()
            print(temp_loss)
            for i in range(len(temp_loss)):
                loss_excel[i] = loss_excel[i] + temp_loss[i]
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    train_loss = train_loss / len(train_data_loader)
    val_loss = val_loss / len(val_data_loader)
    print('epoch %d train loss = %.5f' % (epoch + 1, train_loss))
    print('epoch %d val loss = %.5f' % (epoch + 1, val_loss))
    excel_val_line = write_excel(sheet=sheet_val,
                                 data_type='val',
                                 line=excel_val_line,
                                 epoch=epoch,
                                 itr=False,
                                 loss=[loss_excel, val_loss, train_loss],
                                 weight=False)
    f.save(excel_save)
    if val_loss < min_loss:
        min_loss = val_loss
        min_epoch = epoch
        torch.save(net, save_model)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
print('Train is Done!')
