"""
实现思路1：重现 import_data 的过程，完全模拟网络的输入过程，这样需要重构的代码比较多
实现思路2：直接调用 obj_detector 模块，查看真实数据中无法有效输出的data，下面采用该思路实现
"""
from pre_process.obj_detector import obj_detector


if __name__ == "__main__":

    data_size = 1000
    count = 0
    print("checking...")

    for index in range(1, data_size+1):
        try:
            obj = obj_detector(index)
            obj.cal_box(False)
            obj.visual(show_img=False, show_grid=False, rectangle=False, save=False)

        except:
            count += 1
            print("broken index: ", index)
    print("total broken data is: ", count)
    print("you should delete this number in trainval.txt")


