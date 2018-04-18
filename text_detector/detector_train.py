import os
import argparse
import datetime
import tensorflow as tf
from text_detector.detect_net import config as cfg
from text_detector.detect_net.yolo_net import YOLONet
from text_detector.utils.timer import Timer
from text_detector.utils.import_data import text_detect_obj
from text_detector.utils.logging import yolo_log
from pre_process.report import send_email


slim = tf.contrib.slim

# 这部分代码主要实现的是对已经构建好的网络和损失函数利用数据进行训练
# 在训练过程中，对变量采用了指数平均数（exponential moving average (EMA)）
# 来提高整体的训练性能。同时，为了获得比较好的学习性能，对学习速率同向进行
# 了指数衰减，使用了 exponential_decay 函数来实现这个功能。

# 在训练的同时，对我们的训练模型(网络权重)进行保存，这样以后可以直接进行调
# 用这些权重；同时，每隔一定的迭代次数便写入 TensorBoard，这样在最后可以观察整体的情况。


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER

        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.save_cfg()

        # tf.get_variable 和tf.Variable不同的一点是，前者拥有一个变量检查机制，
        # 会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，
        # TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo_text_detect.ckpt')

        # tf 中使用 summary 来可视化我们的数据流，最终使用一个merge_all 函数来管理所有的摘要
        # tf 中 summary 的计算需要 feed 数据
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()

        # 学习率衰减方案
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')

        # 指定模型优化方案：使用GD求解
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)

        # 基于模型，损失函数和优化方案进行优化
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        # if cfg.WEIGHTS_FILE is not None:
        if self.weights_file is not None:
            print(self.weights_file)
            log_str = 'Restoring weights from: ' + self.weights_file
            print(log_str)
            yolo_log(log_str)

            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        # 设定计时器
        train_timer = Timer()
        load_timer = Timer()

        # 总共迭代 max_iter 次
        for step in range(1, self.max_iter + 1):

            load_timer.tic()

            # 获取输入的 img 和 label，每次调用返回不同的数据对
            images, labels = self.data.get()
            load_timer.toc()

            # 将输入数据转化为 feed_dict 形式输入网络
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            # 检查点：每迭代 summary_iter=100，输出当前的迭代位置，损失以及相关状态
            if step % self.summary_iter == 0:

                # 每迭代10个 summary_iter=1000，对模型配置和参数进行存档
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()

                    # 执行一个会话，获取当前阶段的运行情况，损失，以及优化损失
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    # use format to remake the log info.
                    log_str = '''
                    {} Epoch: {}, 
                    Step: {}, 
                    Learning rate: {},
                    Loss: {:5.3f}\n
                    Speed: {:.3f}s/iter,
                    Load: {:.3f}s/iter,
                    Remain: {}
                    '''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter)
                    )
                    print(log_str)
                    yolo_log(log_str)

                else:   # 只进行检查，不存档
                    train_timer.tic()
                    # 未到10倍存档节点，则继续训练网络，进行两种工作：计算 summary 和优化损失
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                # 写入日志
                self.writer.add_summary(summary_str, step)

            else:   # 不检查也不存档，只进行训练
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            # 存档点：这里与检查点是分开的，每进行 save_iter 步对模型配置和参数进行存档
            if step % self.save_iter == 0:

                # 提示信息
                log_str = '{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir)
                print(log_str)
                yolo_log(log_str)

                # save 操作
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):
        # 存储模型参数，开始训练时就执行一遍
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):

    cfg.DATA_PATH = data_dir
    cfg.CACHE_PATH = os.path.join(cfg.DATA_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.DATA_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.DATA_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    # parser: convert input message into class or data structs
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_text_detect.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()

    data_set = text_detect_obj('train')

    solver = Solver(yolo, data_set)

    print('Start training ...')
    yolo_log('Start training ...')
    solver.train()
    print('Done training.')
    yolo_log('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0

    main()
