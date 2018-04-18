"""
func: 清洗数据集，将label中的标记项删除，只输出每个box的8个坐标值
tips: because ',' in the labels, so use re deal with the file rather than pandas.
speed: 22.513817s/1K-pic&labels

"""

import os
import re
import shutil
import cv2
from pre_process.travel_all_file import file_list
import pre_process.config as cfg
from pre_process.timer import timer


def trans_img(raw_img, dst_img):

    # read img
    img = cv2.imread(raw_img)

    # code img: maybe improve the quality, some raw pic can't open, maybe change the coding.

    # restore the img  / with new code
    cv2.imwrite(dst_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #shutil.copyfile(raw_img, dst_img)


def trans_label(raw_label, dst_label):
    with open(raw_label, encoding='utf-8') as raw_file:
        with open(dst_label, 'a', encoding='utf-8') as dst_file:

            while True:
                line = raw_file.readline()
                if not line:
                    break
                parser_line(line, dst_file)
                pass


def parser_line(line, dst_file):
    # or count the 7th ','
    buf = re.findall(r"\d+\.?\d*", line)[0:8]
    obj = buf[0] + ',' + buf[1] + ',' + buf[2] + ',' + buf[3] + ',' + buf[4] + \
        ',' + buf[5] + ',' + buf[6] + ',' + buf[7] + '\n'

    dst_file.write(str(obj))


if __name__ == '__main__':

    raw_img_path = os.path.join(cfg.RAW_DATA_PATH, cfg.RAW_IMG_PATH)
    raw_label_path = os.path.join(cfg.RAW_DATA_PATH, cfg.RAW_LABEL_PATH)

    dst_img_path = os.path.join(cfg.DATA_PATH, cfg.IMG_PATH)
    dst_label_path = os.path.join(cfg.DATA_PATH, cfg.LABEL_PATH)

    print('raw data dir: ', raw_label_path)
    print('convert data dir: ', dst_label_path)

    file_list = file_list(raw_label_path)

    count = 1

    timer = timer()
    timer.start()

    if not file_list is None:
        for item in file_list:

            # raw_data
            raw_img = raw_img_path + item + '.jpg'
            raw_label = raw_label_path + item + '.txt'

            # dst_data
            dst_index = '%06d' % count
            dst_img = dst_img_path + dst_index + '.jpg'
            dst_label = dst_label_path + dst_index + '.txt'

            # trans
            # trans_img(raw_img, dst_img)
            # trans_label(raw_label, dst_label)

            count += 1
    else:
        print('data_dir is empty.')

    print('convert file: ', count)

    timer.stop()
