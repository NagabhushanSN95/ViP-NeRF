# Shree KRISHNAya Namaha
# Abstract class
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import abc

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class LearningRateDecayerParent:
    @abc.abstractmethod
    def get_updated_learning_rate(self, iter_num):
        pass
