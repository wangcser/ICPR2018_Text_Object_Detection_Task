import pre_process.config as cfg
import pandas as pd
import numpy as np


def create_index():
    """
    if some data broken, you can just remove it from index file to avoid using it.
    :return: data index
    """
    index_size = 1000
    index_path = cfg.INDEX_PATH
    broken_index_path = cfg.BROKEN_INDEX_PATH

    check_list = []

    with open(broken_index_path) as cf:
        for line in cf:
            check_list.extend(line.strip().split('\n'))

    count = 0

    with open(index_path, 'a') as wf:
        for i in range(1, index_size + 1):

            if str(i) in check_list:
                continue

            index = '%06d' % i
            wf.write(index + '\n')
            count += 1

    print('index generate: ', count)
    return count


if __name__ == "__main__":

    create_index()
