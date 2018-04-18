import os

#
# path and dataset parameter
#
# 超参数配置：第一部分是使用到的数据相关参数，包括数据路径预训练权重等相关内容

DATA_PATH = 'data'

CACHE_PATH = os.path.join(DATA_PATH, 'cache')

OUTPUT_DIR = os.path.join(DATA_PATH, 'output')

WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')

# in this you can use fine-tune weights to start your train
WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_text_detector.ckpt')

CLASSES = []

"""
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']    # 20 class
"""

# 是否对样本图像进行flip（水平镜像）操作
FLIPPED = False


#
# model parameter
#
# 这部分主要是模型参数
# 图像size
IMAGE_SIZE = 448
# 网格 size
CELL_SIZE = 7
# 每个 cell 中 bounding box 数量
BOXES_PER_CELL = 2
# 权重衰减相关参数
ALPHA = 0.1

DISP_CONSOLE = False
# 权重衰减的相关参数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#
# 训练过程中的相关参数
GPU = '0'
# 学习速率
LEARNING_RATE = 0.0001
# 衰减步数
DECAY_STEPS = 30000
# 衰减率
DECAY_RATE = 0.1

STAIRCASE = True
# batch_size初始值为45
BATCH_SIZE = 16
# 最大迭代次数 default is 15000
MAX_ITER = 2000
# 日志记录迭代步数 default is 10
SUMMARY_ITER = 10
# 原始为每1000个样本存档一次权重，每100个样本输出一次网络情况，这样会消耗5GB的空间，现在改为每5000存档一次
# default is 1000
SAVE_ITER = 100


#
# test parameter
#
# 测试时的相关参数
# 阈值参数 default is 0.2
THRESHOLD = 0.001
# IoU 参数
IOU_THRESHOLD = 0.5