import os
import pre_process.config as cfg


def file_list(file_dir):

    if not os.path.exists(file_dir):
        print('file_dir is not exist.')
        return

    file_name_list = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_name_list.append(os.path.splitext(file)[0])
    return file_name_list


if __name__ == '__main__':

    data_path = cfg.DATA_PATH
    label_path = cfg.LABEL_PATH

    dir_path = os.path.join(data_path, label_path)
    name_list = file_list(dir_path)  # with out order.

    print(name_list)

    print('file in list, len: ', len(name_list))
