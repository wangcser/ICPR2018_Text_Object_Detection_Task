import cv2
import pre_process.config as cfg


def show_img(name):

    """
    param img_name:
    return: img info.
    notice: resize and remap ops will cut the quality of img.

    """
    image = cv2.imread(name)
    raw_size = image.shape

    image = cv2.resize(image, (448, 448))
    resize = image.shape

    # 转换色彩空间为 RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # change the pixel num from (0-255) to (0-1)
    image = (image / 255.0) * 2.0 - 1.0
    # cv2.imshow('resize', image)
    # cv2.waitKey()

    print('img type: ', type(image))
    print('raw size:', raw_size, 're size: ', resize)


if __name__ == '__main__':

    data_path = cfg.DATA_PATH
    img_path = cfg.IMG_PATH
    label_path = cfg.LABEL_PATH  #
    img_path = '/sorted_image_1000/'
    img_name = '000001.jpg'

    show_img(data_path + img_path + img_name)
