# Shree KRISHNya Namaha
# Some common utilities
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import json
from pathlib import Path
from typing import List

import pandas


def start_matlab_engine():
    import matlab.engine

    print('Starting MatLab Engine')
    matlab_engine = matlab.engine.start_matlab()
    print('MatLab Engine active')
    return matlab_engine
