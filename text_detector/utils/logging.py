import time


def yolo_log(log_data):
    t = time.localtime()

    time_stamp = str(t.tm_mon) + "." + str(t.tm_mday)
    # + "-" + str(t.tm_hour) + ":" + str(
    # t.tm_min) + ":" + str(t.tm_sec)

    path = '/home/super/PycharmProjects/ICPR2018_text_detection/text_detector/log'
    f = open(path + '/' + time_stamp + '-' + "yolo_log.txt", "a")

    f.write(log_data + "\n")
    f.close()


if __name__ == '__main__':
    yolo_log('hello')
