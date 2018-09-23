# -*- coding: utf-8 -*-

from config import Config
from cycleGAN_Unet import  CycleGAN_Unet

if __name__ == "__main__":
    config = Config()
    model = CycleGAN_Unet(config)
    if config.RESUME_TRAIN:
        model.resume_train()
    else:
        model.train()