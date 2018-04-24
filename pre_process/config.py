"""
the config info. in pre process.
"""
import os


# project path
PROJECT_PATH = os.path.abspath(os.path.dirname(os.getcwd()))
DATA_PATH = os.path.join(PROJECT_PATH, 'text_detector/data/train_data/')
IMG_PATH = os.path.join(DATA_PATH, 'sorted_image_9000/')
LABEL_PATH = os.path.join(DATA_PATH, 'sorted_txt_9000/')
# raw data path
RAW_DATA_PATH = '/media/super/Dev Data/Data Set & Weight/ICPR_text_train/train_9000/'
RAW_IMG_PATH = os.path.join(RAW_DATA_PATH, 'image_9000/')
RAW_LABEL_PATH = os.path.join(RAW_DATA_PATH, 'txt_9000/')

""" use to generate test data.
# project path
PROJECT_PATH = os.path.abspath(os.path.dirname(os.getcwd()))
DATA_PATH = os.path.join(PROJECT_PATH, 'text_detector/data/test_data/')
IMG_PATH = os.path.join(DATA_PATH, 'sorted_image_1000/')
LABEL_PATH = os.path.join(DATA_PATH, 'sorted_txt_1000/')
# raw data path
RAW_DATA_PATH = '/media/super/Dev Data/Data Set & Weight/ICPR_text_train/train_1000/'
RAW_IMG_PATH = os.path.join(RAW_DATA_PATH, 'image_1000/')
RAW_LABEL_PATH = os.path.join(RAW_DATA_PATH, 'txt_1000/')
"""

# save path
SAVE_PATH = os.path.join(PROJECT_PATH, 'pre_process/markdown_resource/')

# index path
INDEX_PATH = os.path.join(DATA_PATH + 'index/trainval.txt')
BROKEN_INDEX_PATH = os.path.join(DATA_PATH + 'index/broken_index.txt')

# report
FROM_EMAIL = '502612842@qq.com'
FROM_EMAIL_TOKEN = 'bvbekgpwcrslbgha'
TO_EMAIL = 'wangcser@qq.com'
SUBJECT = 'Text Detection Task Running Report'
CONTENT = '''
    下方为项目目前运行情况：
    '''