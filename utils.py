# -*- coding: utf-8 -*-

from glob import glob
import random

import scipy.misc
import numpy as np

def data_generator(path, batch_size):
    file_list = glob(path)
    idx = 0
    list_length = len(file_list)
    while True:
        batch = []
        for _ in range(batch_size):
            if idx >= list_length:
                idx = 0
                random.shuffle(file_list)
            batch.append(file_list[idx])
            idx += 1
        yield batch

def set_trainable(model, prefix_list, trainable=False):
    for prefix in prefix_list:
        for layer in model.layers:
            if layer.name.startswith(prefix):
                layer.trainable = trainable
    return model

def get_image(file_path, input_hw):
    image = imread(file_path)
    cropped = crop(image)
    resized = scipy.misc.imresize(cropped, [input_hw, input_hw])
    return np.array(resized)/127.5 - 1 #0→255を-1→1に変換


def imread(file_path):
    return scipy.misc.imread(file_path).astype(np.float32)


def crop(image):
    h, w, c = image.shape
    if h >= w:
        crop_wh = w
        sub = int((h - w) // 2)
        trimmed = image[sub:sub+crop_wh, :, :]
    else:
        crop_wh = h
        sub = int((w - h) // 2)
        trimmed = image[:, sub:sub+crop_wh, :]
    return trimmed


def output_sample_image(path, combine_image):
    image = (combine_image+1) * 127.5
    scipy.misc.imsave(path, image.astype(np.uint8))