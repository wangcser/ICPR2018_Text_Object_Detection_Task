import cv2
import pandas as pd
from PIL import ImageDraw
import pre_process.config as cfg
from pre_process.draw_box import draw_boxes


class obj_detector:

    def __init__(self, img_index):

        self.dst_img_path = cfg.IMG_PATH
        self.dst_label_path = cfg.LABEL_PATH

        self.index = '%06d' % img_index

        self.img = self.dst_img_path + self.index + '.jpg'
        self.label = self.dst_label_path + self.index + '.txt'

        self.IMG_SIZE = 448
        self.CELL_SIZE = 28

    def load_data(self):

        im = cv2.imread(self.img)
        # pandas data-frame, 4 points each box
        text_point = pd.read_csv(self.label, header=None, sep=',')

        # print('load img from: ', self.img)
        return im, text_point

    def visual(self, show_img=True, show_grid=False, rectangle=True, save=False):

        img = draw_boxes(self.img, self.label, rectangle=rectangle)

        if show_grid:   # draw grid on the raw img.
            # img.resize((self.IMG_SIZE, self.IMG_SIZE), Image.ANTIALIAS)
            img_height, img_width = img.size
            x_step = int(img_height / self.CELL_SIZE)
            y_step = int(img_width / self.CELL_SIZE)

            draw_grid = ImageDraw.Draw(img)

            for x_step in range(0, img_height, x_step):
                draw_grid.line(((x_step, 0), (x_step, img_width)), fill='black')

            for y_step in range(0, img_width, y_step):
                draw_grid.line(((0, y_step), (img_height, y_step)), fill='black')

        if save:
            img.save(cfg.SAVE_PATH + 'gt_result_grid_rect.jpg')  # 保存图片

        if show_img:
            img.show()

    def show_central(self):
        pass

    def cal_box(self, output_result=True):

        im, text_point = self.load_data()

        # this way cal the ratio in width and height
        h_ratio = 1.0 * 448 / im.shape[0]  # height
        w_ratio = 1.0 * 448 / im.shape[1]  # width

        count = 0

        for idx, row in text_point.iterrows():
            # index是一个numpy.int64的类型
            # row是一个Series类型，它的index是data的列名

            point = row.loc[range(8)].tolist()  # 依次读取八个点的数据
            x = [point[i] for i in [0, 2, 4, 6]]
            y = [point[i] for i in [1, 3, 5, 7]]
            # point = [(a, b) for a, b in zip(x, y)]

            # reset the co-ordinary prevent the cell overflow
            x1 = max(min((float(min(x)) - 1) * w_ratio, 448 - 1), 0)
            y1 = max(min((float(min(y)) - 1) * h_ratio, 448 - 1), 0)
            x2 = max(min((float(max(x)) - 1) * w_ratio, 448 - 1), 0)
            y2 = max(min((float(max(y)) - 1) * h_ratio, 448 - 1), 0)

            # lower strip use to clean the string.

            # nn use the central point x, y and width, height
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]

            # cell_id of obj detected. use central point to calculate the cell position.
            x_ind = int(boxes[0] * self.CELL_SIZE / 448)
            y_ind = int(boxes[1] * self.CELL_SIZE / 448)
            count += 1

            if output_result:
                # cell in 0-self.CELL_SIZE-1
                print('obj in cell: ', x_ind, y_ind)    # tells that the cell is too large

        if output_result:
            print('obj num:' + str(count))


if __name__ == '__main__':

    index = 1   # int num.

    obj = obj_detector(index)
    obj.cal_box(False)
    obj.visual(show_img=True, show_grid=True, rectangle=True, save=True)
