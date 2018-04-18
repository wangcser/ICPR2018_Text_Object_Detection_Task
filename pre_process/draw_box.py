import os
import pandas as pd
from PIL import Image, ImageDraw
from pre_process.travel_all_file import file_list
import pre_process.config as cfg


def draw_boxes(file_img, file_label, rectangle=False):
    #
    # input:pic and label
    # output: pic with label in raw format and raw size
    #

    if not os.path.exists(file_img):
        print('can\'t find img.')
        return
    if not os.path.exists(file_label):
        print('can\'t find label.')
        return

    img = Image.open(file_img)
    draw = ImageDraw.Draw(img)
    # this method can be wrong when data is very large.
    text_point = pd.read_csv(file_label, header=None, sep=',')

    for idx, row in text_point.iterrows():
        point = row.loc[range(8)].tolist()  # 依次读取八个点的数据
        x = [point[i] for i in [0, 2, 4, 6]]
        y = [point[i] for i in [1, 3, 5, 7]]
        point = [(a, b) for a, b in zip(x, y)]

        if rectangle:   # 如果要画长方形 rectangle = True
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
            draw.rectangle((x_min, y_min, x_max, y_max), outline=(0, 0, 255))
        else:   # 画多边形
            draw.polygon(point, outline=(0, 128, 255))

    return img


if __name__ == '__main__':

    data_path = cfg.DATA_PATH
    img_path = cfg.IMG_PATH
    label_path = cfg.LABEL_PATH

    # load file from dir.
    # if dir is empty, print: not exist.
    dir_path = os.path.join(data_path, label_path)
    file_name_list = file_list(dir_path)

    # get a file name from file list(with out the path info.).
    if not file_name_list is None:

        # you can also name the file_index.
        # file_name = file_name_list[1]
        file_index = '000001'

        file_img = os.path.join(data_path, img_path, '%s.jpg' % file_index)  # 图片的地址
        file_label = os.path.join(data_path, label_path, '%s.txt' % file_index)  # 描述文件的地址

        img = draw_boxes(file_img, file_label, True)
        img.show()

        img.save(cfg.PROJECT_PATH + '/pre_process/markdown_resource/' + '/demo_rec.jpg')  # 保存图片
    else:
        pass

