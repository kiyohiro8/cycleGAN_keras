# -*- coding: utf-8 -*-


class Config():

    def __init__(self):
        #TODO Make comments at each parameters
        self.EPOCH = 100
        self.ITER_PER_EPOCH = 1000
        self.BATCH_SIZE = 2
        self.LEARNING_RATE = 0.002
        self.BETA_1 = 0.5
        self.LAMBDA = 10

        self.RESULT_DIR = "./result/"
        self.DATA_DIR = "./data/"

        self.INPUT_SHAPE = (256, 256, 3)
        self.DATA_EXT = "*.jpg"
        self.DATASET_A = "FriedRice"
        self.DATASET_B = "Paella"
