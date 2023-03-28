# Shree KRISHNAya Namaha
# 
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path

from lr_decayers.LearningRateDecayerParent import LearningRateDecayerParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class NeRFLearningRateDecayer(LearningRateDecayerParent):
    def __init__(self, configs: dict):
        self.configs = configs
        self.lr_init = self.configs['optimizer']['lr_initial']
        self.decay_rate = 0.1
        self.decay_steps = self.configs['optimizer']['lr_decay'] * 1000
        return

    def get_updated_learning_rate(self, iter_num):
        new_lr = self.lr_init * (self.decay_rate ** (iter_num / self.decay_steps))
        return new_lr
