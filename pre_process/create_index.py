import pre_process.config as cfg


def create_index():
    """
    if some data broken, you can just remove it from index file to avoid using it.
    :return: data index
    """
    file_name = '/home/super/PycharmProjects/ICPR2018_text_detection/text_detector/data/train_data/index/trainval.txt'
    with open(file_name, 'a') as wf:

        for i in range(1, 1001):
            index = '%06d' % i
            # wf.write(index + '\n')

    wf.close()


if __name__ == "__main__":

    create_index()
