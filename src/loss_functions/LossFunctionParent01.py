# Shree KRISHNAya Namaha
# Abstract parent class
# Extended from LossFunctionParent01.py. Can compute loss for validation images as well
# Author: Nagabhushan S N
# Last Modified: 15/02/2022

import abc


class LossFunctionParent:
    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict, training: bool = True):
        pass
