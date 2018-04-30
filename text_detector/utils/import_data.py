import pickle
import copy
from cv2 import *
import numpy as np
import pandas as pd
import text_detector.detect_net.config as cfg


class text_detect_obj(object):

    def __init__(self, phase, rebuild=False):
        # rebuild use to update the data-label cache.

        self.data_path = cfg.DATA_PATH
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.flipped = cfg.FLIPPED

        # train or test stage
        self.phase = phase
        # update the label cache
        self.rebuild = rebuild

        # feed tools
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
        # 将每个像素点的值压缩到 -1~+1
        image = (image / 255.0) * 2.0 - 1.0
        # 是否做图像翻转--增强操作
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):

        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = \
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 - \
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp

        # 将序列元素随机排序
        np.random.shuffle(gt_labels)

        self.gt_labels = gt_labels

        return gt_labels

    def load_labels(self):

        # cached the gt labels
        # text_detect_train_gt_labels.pkl
        cache_file = os.path.join(
            self.cache_path, 'text_detect_' + self.phase + '_gt_labels.pkl')

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # 如果存在 gt_label 的缓存文件 而且不要求完全重新学习的话，就从缓存中获取 labels
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if self.phase == 'train':
            # 返回图片的 index
            txtname = os.path.join(
                self.data_path, 'train_data', 'index', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'test_data', 'index', 'test.txt')
        # ???
        with open(txtname, 'r') as f:
            # 获取一个文件 index 000023
            # 组装成一个 list
            self.image_index = [x.strip() for x in f.readlines()]

        # 不从缓存读取 gt_label，则单独读取
        gt_labels = []

        for index in self.image_index:
            print('index', index)

            # 获取当前的 label 和 obj 的数量
            label, num = self.load_label(index)

            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'train_data', 'sorted_image_9000', index + '.jpg')

            # 将所有需要的信息组装成一个 dict
            # img_path, label, flipped_flags
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})

        # 获取 gt_labels 的同时也将其做成缓存文件
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_label(self, index):

        # import img
        imname = os.path.join(self.data_path, 'train_data', 'sorted_image_9000', index + '.jpg')
        im = cv2.imread(imname)

        # this way calu the ratio in width and height
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        # import labels
        label = np.zeros((self.cell_size, self.cell_size, 5))

        filename = os.path.join(self.data_path, 'train_data', 'sorted_txt_9000', index + '.txt')

        # pandas data-frame, 4 points each box
        text_point = pd.read_csv(filename, header=None, sep=',')
        count = 0

        for idx, row in text_point.iterrows():
            # index是一个numpy.int64的类型
            # row是一个Series类型，它的index是data的列名
            # print('idx:', idx)
            # print('row:', row)

            # 依次读取八个点的数据，可能会出现精度问题
            point = row.loc[range(8)].tolist()
            x = [point[i] for i in [0, 2, 4, 6]]
            y = [point[i] for i in [1, 3, 5, 7]]

            # reset the co-ordinary prevent the cell overflow
            # 可能是导入的有的模块中存在min，max方法导致覆盖了python 自带的方法！！！
            # 使用np.min 可以解决……

            x1 = max(min((float(np.min(x)) - 1) * w_ratio, 448 - 1), 0)
            y1 = max(min((float(np.min(y)) - 1) * h_ratio, 448 - 1), 0)
            x2 = max(min((float(np.max(x)) - 1) * w_ratio, 448 - 1), 0)
            y2 = max(min((float(np.max(y)) - 1) * h_ratio, 448 - 1), 0)

            # 本来 min(x)应该输出一个值, 但是现在输出的是一个4-d list
            # 只能这么处理
            x1 = x1[0][0]
            x2 = x2[0][0]
            y1 = y1[0][0]
            y2 = y2[0][0]

            # lower strip use to clean the string.

            # nn use the central point x, y and width, height
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]

            # cell_id of obj detected. use central point to calculate the cell position.
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)

            count += 1

            # in cv2, image store in height, width, channel
            label[y_ind, x_ind, 0] = 1  # have obj
            label[y_ind, x_ind, 1:5] = boxes  # box of obj

        return label, count
