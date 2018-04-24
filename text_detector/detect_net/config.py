import os

#
# path and data set parameter
#
# 超参数配置：第一部分是使用到的数据相关参数，包括数据路径预训练权重等相关内容

DATA_PATH = 'data'

CACHE_PATH = os.path.join(DATA_PATH, 'cache')

OUTPUT_DIR = os.path.join(DATA_PATH, 'output')

WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')

# choose use fine tuning weights or not.
WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'yolo_text_detect.ckpt-30000')

# data set enhancement.
# 是否对样本图像进行flip（水平镜像）操作
FLIPPED = False


#
# model parameter
#
# 这部分主要是模型参数
# 图像size
IMAGE_SIZE = 448
# 网格 size
CELL_SIZE = 28
# 每个 cell 中 bounding box 数量
BOXES_PER_CELL = 2
# 权重衰减相关参数
ALPHA = 0.1
# drop out probability
DROP_OUT_PROB = 0.5

# 是否在控制台显示相关输出
DISP_CONSOLE = False

# loss 中各部分的权重
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
# CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#
# GPU
GPU = '0'
# 学习速率 default is 0.0001
LEARNING_RATE = 0.0001
# 衰减步数 default is 30000
DECAY_STEPS = 30000
# 衰减率 default is 0.1
DECAY_RATE = 0.1
# ?
STAIRCASE = True
# batch_size default is 45
BATCH_SIZE = 16
# 最大迭代次数 default is 15000, 100,000 need about 21 hour.
# i choose 30,000 will cost about 7 hours.
MAX_ITER = 10000
# 日志记录迭代步数, means logs num. default is 10
SUMMARY_ITER = 10
# 原始为每1000个样本存档一次权重，每100个样本输出一次网络情况，这样会消耗5GB的空间，现在改为每5000存档一次
# default is 1000
SAVE_ITER = 20000


#
# test parameter
#
# 测试时的相关参数
# 阈值参数 box confidence default is 0.2
THRESHOLD = 0.02
# IoU 参数
IOU_THRESHOLD = 0.5
