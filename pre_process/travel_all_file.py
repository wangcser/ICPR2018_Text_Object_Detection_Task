import os
import pre_process.config as cfg


def file_list(travel_dir, show=True):

    if not os.path.exists(travel_dir):
        print('travel_dir is not exist.')
        return

    file_name_list = []

    for root, dirs, files in os.walk(travel_dir):
        for file in files:
            file_name_list.append(os.path.splitext(file)[0])

    if show:
        file_name_list = sorted(file_name_list)
        print(file_name_list)
    return file_name_list


if __name__ == '__main__':

    label_path = cfg.LABEL_PATH

    name_list = file_list(label_path, show=True)  # with out order.

    print('file in list, len: ', len(name_list))
