# Shree KRISHNAya Namaha
# 
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path

import numpy

from lr_decayers.LearningRateDecayerParent import LearningRateDecayerParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MipNeRFLearningRateDecayer(LearningRateDecayerParent):
    def __init__(self, configs: dict):
        self.configs = configs
        self.lr_init = self.configs['optimizer']['lr_initial']
        self.lr_final = self.configs['optimizer']['lr_final']
        self.num_iters = self.configs['num_iterations']
        self.lr_decay_steps = self.configs['optimizer']['lr_decay_steps']
        self.lr_decay_mult = self.configs['optimizer']['lr_decay_mult']
        return

    def get_updated_learning_rate(self, iter_num):
        if self.lr_decay_steps > 0:
            # A kind of reverse cosine decay.
            decay_rate = self.lr_decay_mult + (1 - self.lr_decay_mult) * numpy.sin(0.5 * numpy.pi * numpy.clip(iter_num / self.lr_decay_steps, 0, 1))
        else:
            decay_rate = 1.
        t = numpy.clip(iter_num / self.num_iters, 0, 1)
        log_lerp = numpy.exp(numpy.log(self.lr_init) * (1 - t) + numpy.log(self.lr_final) * t)
        new_lr = decay_rate * log_lerp
        return new_lr
