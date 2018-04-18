import os
import pickle
import copy
import cv2
import numpy as np
import pandas as pd
import text_detector.yolo_net.config as cfg


class text_detect_obj(object):

    def __init__(self, phase, rebuild=False):
        self.data_path = os.path.join(cfg.DATA_PATH, 'train_data')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES  # should be empty
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))   # should be 0
        self.flipped = cfg.FLIPPED

        # current data set label?
        self.phase = phase
        # ?
        self.rebuild = rebuild
        # ?
        self.cursor = 0
        self.epoch = 1
        # ground true labels
        self.gt_labels = None

        # this way input data
        self.prepare()

    def get(self):

        # the dim is 16*448*448*3
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        # 16*7*7*5
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 5))

        count = 0

        # 随机镜像图片
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):

        image = cv2.imread(imname)

        # 在这里调整图片的大小, resize 不会去保证图像的比例，也就是说不填充，只拉伸
        # 目前看来输入网络时是不考虑label的变换的
        image = cv2.resize(image, (self.image_size, self.image_size))

        # 转换色彩空间为 RGB，这里看不懂，调整通道的意义在哪里？
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        # 是否做图像翻转--增强操作
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        #
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels

        print('gt_labels')
        print(gt_labels)
        return gt_labels

    def load_labels(self):
        # cached the gt labels
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':

            # 返回图片的 index
            txtname = os.path.join(
                self.data_path, 'train_data', 'index', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'test_data', 'index', 'test.txt')
        # ???
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        # a list store dict maybe in 16
        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # import img
        imname = os.path.join(self.data_path, 'sorted_image_1000', index + '.jpg')
        im = cv2.imread(imname)

        # this way calu the ratio in width and height
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        # import labels
        label = np.zeros((self.cell_size, self.cell_size, 5))
        filename = os.path.join(self.data_path, 'sorted_txt_1000', index + '.txt')
        # pandas data-frame, 4 points each box
        text_point = pd.read_csv(filename, header=None, sep=',')
        count = 0

        for idx, row in text_point.iterrows():
            # index是一个numpy.int64的类型
            # row是一个Series类型，它的index是data的列名
            # print('idx:', idx)
            # print('row:', row)

            point = row.loc[range(8)].tolist()  # 依次读取八个点的数据
            x = [point[i] for i in [0, 2, 4, 6]]
            y = [point[i] for i in [1, 3, 5, 7]]

            # reset the co-ordinary prevent the cell overflow
            x1 = max(min((float(min(x)) - 1) * w_ratio, 448 - 1), 0)
            y1 = max(min((float(min(y)) - 1) * h_ratio, 448 - 1), 0)
            x2 = max(min((float(max(x)) - 1) * w_ratio, 448 - 1), 0)
            y2 = max(min((float(max(y)) - 1) * h_ratio, 448 - 1), 0)

            # lower strip use to clean the string.

            # nn use the central point x, y and width, height
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # print(boxes)

            # cell_id of obj detected. use central point to calculate the cell position.
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)

            count += 1

            # in cv2, image store in height, width, channel
            label[y_ind, x_ind, 0] = 1  # have obj
            label[y_ind, x_ind, 1:5] = boxes  # box of obj

        return label, count

if __name__ == "__main__":

    tmp = text_detect_obj('train')
    tmp.get()
