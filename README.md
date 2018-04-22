# ICPR2018_text_detection

## Introduction
This subject aims at the competition in ali-tianchi, to solve the problem of text detection in item-sell image from ali environment.

For object detection task, there are many methods to deal with. mainly in following three ways:

1. traditional manual obj. feature based methods
2. Region Proposal methods: R-CNN and so on
3. end-to-end methods: YOLO, SSD 
4. segment methods

In the context of the Deep Learning, we team have learned methods 2-4, and decide to solve this problem in two ways. Below is a short conclusion on those methods:

- traditional methods requires lots of expects experience. 
- R-CNN have lots of modules to adjust and fine-tune the net place a great challenge for a little team. But R-CNN may play better on small and clustering obj.
- YOLO give a NN-style methods on this task, and may be easy to design and adjust, so we choose this method to practice. But YOLO plays poor on small obj. we need fix this problem with YOLOv2 or other methods.
- segment give us the promising result in boxing and small obj, so we also use the segment methods.

The biggest challenge in this competition is that there are many small objs and many of them are in clustering(OMG).

using YOLO may work fast but preform poorly on this task. and segment method may play promisingly, our team decide separate our people into three part. part A use YOLO method, part B use segment method, part C do pre process and post process.

## TO DO
- [x] realize draw_box module in post process
- [x] pre process dataset(including img and labels, some img can't use, just pick them out)
- [x] learn YOLO net and other obj. detection methods
- [x] analysis the feature in dataset(maybe use cluster method, part C do this work.) 
- [x] design detect_net, make it work on train and test
- [x] auto check data validation.
- [x] fix the bug that, the box area can't match the full image.
- [x] re optimise the code. 
- [x] modify the path methods, use the relevant path.
- [x] prepare the data set about 10000
- [ ] mid size train 100,000 samples.
- [ ] add advance tensorflow func. in it, such as loss monitor
- [ ] adjust detect_net, make it preform some right but not good results
- [ ] use small train set training
- [ ] separate the data set to train and test.
- [ ] opt detect_net
- [ ] give a best weights and config

## Project file structure

```
ICPR2018_text_detection
	pre_process
		- draw_box.py: draw ground true result
		- label_input.py: input label 
		- travel_all_file.py: get data index
		- voc_data_test.py: format label data into voc style
	text_detector
		data
			cache: ground true label cache.
			output: the weight in checkpoint and tf-events
			train_data: data_index, sorted_img, sorted_txt
			weights: fine-tune net weights
		utils
			- import_data.py
			- timer.py
			- logging.py
		detect_net
			- config.py: define the hiper args in the nn.
			- yolo_net.py: use YOLO structure and defined the loss
		- detector_train.py: train the net, output the weights
		- detector_test.py: test the net, output labels.
	yolo_tensorflow - reference for yolo realize in tf.
	post_process
		- draw_box.py
		- prefromance.py: calu the score for the nn.
		- report: in wechat, qq, email, ...
		
	README.md - it's me!
```

