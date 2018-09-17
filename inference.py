# -*- coding: utf-8 -*-

from config import Config
from cycleGAN import  CycleGAN

if __name__ == "__main__":
    config = Config()
    model = CycleGAN(config)
    model.inference(model_path=config.MODEL_FILE_PATH,target_dir=config.TARGET_DIR)