import numpy as np
import tensorflow as tf
import text_detector.detect_net.config as cfg


# use tf slim module.
slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, is_training=True):
        #
        # 使用cfg文件对网络超参数进行初始化
        #

        # 这两个参数不再需要
        # self.num_class = 0  # 0

        self.image_size = cfg.IMAGE_SIZE    # 448

        self.cell_size = cfg.CELL_SIZE  # S=7
        self.boxes_per_cell = cfg.BOXES_PER_CELL    # B=2

        # 输出大小 7*7*10
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.boxes_per_cell * 5)  # S*S*(B*5)
        # 每个cell的像素大小
        self.scale = 1.0 * self.image_size / self.cell_size

        # 整体类别信息的张量维度 0, 每个bnd 的大小
        # self.boundary1 = self.cell_size * self.cell_size * self.num_class
        # 整体boundingBox的维度 7*7*2
        # self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        self.boundary = self.cell_size * self.cell_size * self.boxes_per_cell

        # loss func 中的权值，分为4部分
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE

        # self.class_scale = cfg.CLASS_SCALE
        # self.class_scale = 0
        self.coord_scale = cfg.COORD_SCALE

        # 训练过程的超参数
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        # 学习衰减率
        self.alpha = cfg.ALPHA
        self.drop_prob = cfg.DROP_OUT_PROB

        # [0-6]*7*2, 2*7*7 -> 7*2*[0-6], 7*7*2
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        # 输入张量 将图像输入设计为 batch_size*448*448*3 float32 的张量
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

        # 定义网络的输入 batch_size*448*448*3, 输出大小 7*7*10, 学习率, 训练标记
        # DNN 中 logits 指的是未归一化的概率，此时还未进入 softmax
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            # 每个cell的张量 batch_size*S*S*5 此时只输出一个最合适的 boundingBox
            # ground true labels
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5])

            # 对损失层传入数据
            self.loss_layer(self.logits, self.labels)
            # 计算总体的损失
            self.total_loss = tf.losses.get_total_loss()
            # 记录 tf 事件
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,   # tf.image
                      num_outputs,  # 7*7*10
                      alpha,    # learn rate
                      keep_prob=0.5,    # drop out
                      is_training=True,
                      scope='yolo'):    # args work field.

        #
        # 构建网络使用了TF的slim模块
        # 主要的函数有slim.arg_scope slim.conv2d slim.fully_connected slim.dropout ……
        #

        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],    # 确定网络参数的作用范围
                activation_fn=leaky_relu(alpha),    # leaky_ReLU alpha?
                weights_regularizer=slim.l2_regularizer(0.0005),    # 对参数 L2正则化
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)  # 对参数 Gauss 初始化
            ):
                # images = batch_size*448*448*3
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')  # 先做padding

                # input net = X*454*454*3
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                # batch_size*224*224*64
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                # batch_size*112*112*64
                net = slim.conv2d(net, 192, 3, scope='conv_4')  # 不给出stride时，认为是SAME的
                # 112*112*192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                # 56*56*192
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                # 56*56*128
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                # 56*56*256
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                # 56*56*256
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                # 56*56*512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                # 28*28*512
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                # 28*28*256
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                # 28*28*512
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                # 28*28*256
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                # 28*28*512
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                # 28*28*256
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                # 28*28*512
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                # 28*28*256
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                # 28*28*512
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                # 28*28*512
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                # 28*28*1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                # 14*14*1024
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                # 14*14*512
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                # 14*14*1024
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                # 14*14*512
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                # 14*14*1024
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                # CNN output: 16*14*14*1024

                # cache CNN input: 16*14*14*1024
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                # 16*16*1024

                # 16*16*1024
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                # 8*8*1024
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                # 8*8*1024
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                # 8*8*1024 conv stop here.
                # cache CNN output: 16*8*8*1024

                # FC input: 16*1024*8*8
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                # 16*1024*8*8
                net = slim.flatten(net, scope='flat_32')
                # just flatten it. one hot vector in 1,048,576
                net = slim.fully_connected(net, 512, scope='fc_33')
                # after hidden layer: 512, maybe be larger.
                net = slim.fully_connected(net, 4096, scope='fc_34')
                # after hidden layer: 4096
                net = slim.dropout(  # if in test stage, this layer will not work.
                    net, keep_prob=keep_prob, is_training=is_training, scope='dropout_35')
                # 4096 with dropout
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
                # num_outputs: 490 like but not 7*7*10 shape in one hot vector

                # 返回计算结果, 可以只改输出层
                return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):

        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
                : [16, 7, 7, 2 ,4]
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
                : [16, 7, 7, 2 ,4]

          这里的box1可以是真实的box, 也可以是两个bnd中的一个，box2同理
          如果要改 box 数量，那么这个函数就要改
          目前只能计算矩形区域的面积

        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """

        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            # tf.stack matrix join ops.
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            # thus the box must be the square, 修改这里可能能够实现非矩形区域的识别
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection the I in IoU, 2-D vector in [length, width]
            intersection = tf.maximum(0.0, rd - lu)
            # square of I
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]
            # square of U
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        # full IoU
        # 输出时将计算结果压缩到 0-1 区间，代表有物体的概率及其可信度
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        # predicts = 1470 = 7*7*10
        # labels = 490 = 7*7*5

        with tf.variable_scope(scope):

            # predict labels

            # 将预测结果 的 1 ~ 2 转换为每个 bnd 对 obj 的响应值
            # 将 predict 中 16 batch 的所有 bnd_1 到 bnd_2 的值转换为张量形式
            # bnd_2 -> 16*7*7*2 =
            predict_scales = tf.reshape(
                predicts[:, :self.boundary],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])

            # 将预测结果剩余的维度 3 ~ 10 转变为每个 box 的坐标
            # predict_boxes = 16*7*7*(2*4)
            predict_boxes = tf.reshape(
                predicts[:, self.boundary:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # ground true labels
            # 将真实的  labels 转换为相应的矩阵形式
            # labels = 7*7*(5)

            # 每个 cell 中 ground true box 的 obj 的响应值
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])

            # 每个 cell 中 ground true box 的坐标 为 5 维张量
            # boxes = 16*7*7*1(box)*4(coordinator)
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])

            # tile 用于张量复制，将 box 复制 bnd_num 份（广播），计算时直接对 box 和 predict_box 比较即可
            # 并对 box 大小（长和宽）做归一化（predict_box 作为网络的原始输出，是归一化了的）
            # boxes = 16*7*7*2(box)*4/img.size
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size

            # offset = 1*7*7*2 指的是每个ground true box 相对于 cell 的偏移
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])

            # offset = 16*7*7*2 计算 box 中心到 cell 锚点的偏移
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            # offset_tran = 16*7*7*2
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            # shape为 [4, batch_size, 7, 7, 2]
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            # 根据阈值判断IoU情况
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # 非最大抑制，输出每个 cell 中 IoU 最大的 predict_box
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            # 参数中加上平方根是对 w 和 h 进行开平方操作，原因在论文中有说明
            # #shape为(4, batch_size, 7, 7, 2)
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # object_loss 有目标物体存在的损失
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss 没有目标物体时的损失
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss 坐标损失 #shape 为 (batch_size, 7, 7, 2, 1)
            coord_mask = tf.expand_dims(object_mask, 4)
            # shape 为(batch_size, 7, 7, 2, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            # 将所有损失放在一起
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            # 将每个损失添加到日志记录
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
