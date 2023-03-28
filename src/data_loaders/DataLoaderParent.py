# Shree KRISHNAya Namaha
# A parent class for all dataloaders
# Author: Nagabhushan S N
# Last Modified: 29/03/2023
import abc

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DataLoaderParent:
    @abc.abstractmethod
    def load_data(self):
        pass
