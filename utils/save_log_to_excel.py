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


def write_excel(sheet, excel_line, epoch_to_save, train_loss_to_save, validation_loss_to_save, learning_rate):
    # 在train页中保留train的数据，在validation页中保留validation的数据
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间
    sheet.write(excel_line, 0, epoch_to_save + 1)
    sheet.write(excel_line, 1, round(train_loss_to_save, 3))
    sheet.write(excel_line, 2, round(validation_loss_to_save, 3))
    sheet.write(excel_line, 3, str(learning_rate))
    excel_line += 1
    return excel_line


def init_excel():
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('loss', cell_overwrite_ok=True)
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
    row0 = ["epoch", "train", "validation", "LR"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
    return workbook, sheet1
