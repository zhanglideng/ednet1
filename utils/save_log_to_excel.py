import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def write_excel(sheet, data_type, line, epoch, itr, l2_loss, ssim_loss, loss, lr):
    # 在train页中保留train的数据，在validation页中保留validation的数据
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间]
    """
    :param sheet:
    :param data_type:
    :param line:
    :param epoch:
    :param itr:
    :param l2_loss:
    :param ssim_loss:
    :param loss:
    :param lr:
    :return:
    train=["EPOCH", "ITR", "L2_LOSS", "SSIM_LOSS", "LOSS", "LR"]
    val=["EPOCH", "L2_LOSS", "SSIM_LOSS", "LOSS", "PSNR", "SSIM", "LR"]
    """
    if data_type == 'train':
        sheet.write(line, 0, epoch + 1)
        sheet.write(line, 1, itr + 1)
        sheet.write(line, 2, round(l2_loss, 3))
        sheet.write(line, 3, round(ssim_loss, 3))
        sheet.write(line, 4, round(loss, 3))
        sheet.write(line, 5, lr)
    else:
        sheet.write(line, 0, epoch + 1)
        sheet.write(line, 1, round(l2_loss, 3))
        sheet.write(line, 2, round(ssim_loss, 3))
        sheet.write(line, 3, round(loss, 3))
        sheet.write(line, 4, lr)
    return line + 1


def init_excel():
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('train', cell_overwrite_ok=True)
    sheet2 = workbook.add_sheet('val', cell_overwrite_ok=True)
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
    row0 = ["EPOCH", "ITR", "L2_LOSS", "SSIM_LOSS", "LOSS", "LR"]
    row1 = ["EPOCH", "L2_LOSS", "SSIM_LOSS", "LOSS", "PSNR", "SSIM", "LR"]
    for i in range(0, len(row0)):
        print('写入train_excel')
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
    for i in range(0, len(row1)):
        print('写入val_excel')
        sheet2.write(0, i, row1[i], set_style('Times New Roman', 220, True))
    return workbook, sheet1, sheet2
