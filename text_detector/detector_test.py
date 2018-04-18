import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import text_detector.detect_net.config as cfg
from text_detector.detect_net.yolo_net import YOLONet
from text_detector.utils.timer import Timer

# 最后给出test 部分的源码，这部分需要使用我们下载好的 “YOLO_small.ckpt”
# 权重文件，当然，也可以使用我们之前训练好的权重文件。 这部分的主要内容就是
# 利用训练好的权重进行预测，得到预测输出后利用 OpenCV 的相关函数进行画框等
# 操作。同时，还可以利用 OpenCV 进行视频处理，使程序能够实时地对视频流进行
# 检测。因此，在阅读本段程序之前，大家应该对 OpenCV 有一个大致的了解。


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        # those two args should be removed.
        self.classes = cfg.CLASSES
        # self.num_class = len(self.classes)
        self.num_class = 20

        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD

        # below is the offset for each type of data
        self.boundary2 = self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)

        # old version saver
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][0])
            y = int(result[i][1])

            # why w / 2
            w = int(result[i][2] / 2)
            h = int(result[i][3] / 2)

            # use the dui jiao xian draw the box, use the dl, ur point.
            # rectangle(img, dl, ur, box_color[, thickness, line_type, shift])
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)

            # draw a background for the text
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA

            cv2.putText(
                img, ' : %.2f' % result[i][4],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):

        img_h, img_w, _ = img.shape

        # reshape the img
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        # raw result is:
        result = self.detect_from_cvmat(inputs)[0]

        # reshape the labels in true scale.
        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        # reshaped result is:
        return result

    def detect_from_cvmat(self, inputs):
        # net_output is: 1*490
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):

            # input the whole info, output the result.
            results.append(self.interpret_output(net_output[i]))

        # interpret the raw 1*490 result into format result n*5.
        # result is a n*5 matrix, n is the num of box, 5 = 4 + 1
        print(results)
        return results

    # input is: 1*490 in list type.
    # output is n*5
    def interpret_output(self, output):
        '''
        # probs in 7*7*2*20
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))

        # class_probs is: 7*7*0 so this is 0,maybe remove it.
        # self.boundary1 = 0
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        '''

        # scales: 0-97
        # the first 98 nums is the P(obj) for each bnd in cells.
        # change to: 7*7*2
        scales = np.reshape(
            output[0:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))

        # boxes: 98-490
        # the second 392 nums is the bnd axis for each bnd in cells.
        # change to: 7*7*2*4
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))

        # use bnd info cal each box offset.
        # offset: [0-6]*7*2
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)

        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))

        # boxes in: 7*7*2*4
        # rescale the box to img size.
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # rescale the box to img size.
        boxes *= self.image_size

        '''
        # important!!!
        for i in range(self.boxes_per_cell): # 2
            for j in range(self.num_class): # 20

                # there cal the P(class)=P(class|obj)*P(obj)
                # there should modified.
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i]) #

        # cal the P(obj) just remove it .
        for i in range(self.boxes_per_cell):  # 2
            scales[:, :, i]
        '''

        # filter the bnd to only one in each cell.
        filter_mat_probs = np.array(scales >= self.threshold, dtype='bool')
        #filter_mat_probs = np.array(probs >= 0.001, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        #print(filter_mat_boxes)

        # filter box
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        # filter probs
        probs_filtered = scales[filter_mat_probs]

        '''
        # filter classes
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        '''

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]

        probs_filtered = probs_filtered[argsort]

        #classes_num_filtered = classes_num_filtered[argsort]
        #print(classes_num_filtered)

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        #classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [
                    #self.classes[classes_num_filtered[i]],
                    boxes_filtered[i][0],
                    boxes_filtered[i][1],
                    boxes_filtered[i][2],
                    boxes_filtered[i][3],
                    probs_filtered[i]
                ]
            )

        # here print the list of box.
        print('this is interpreted result: ', len(result))
        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()

        # result is:
        result = self.detect(image)

        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        # this is the core in this module
        self.draw_result(image, result)

        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():

    # parser refers to change some data to some formated data.
    parser = argparse.ArgumentParser()

    # raw weights restore methods
    #parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    #parser.add_argument('--weight_dir', default='weights', type=str)

    parser.add_argument('--weights', default="yolo_text_detect.ckpt-2000", type=str)
    parser.add_argument('--weight_dir', default='output/2018_04_14_15_55/', type=str)

    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # build compute graph
    yolo = YOLONet(False)

    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    # input graph and weights get the net
    detector = Detector(yolo, weight_file)

    # detect from image file
    imname = 'test/000003.jpg'

    # pass img to the net, get the prediction
    detector.image_detector(imname)


if __name__ == '__main__':

    # np.set_printoptions(suppress=True)
    main()
