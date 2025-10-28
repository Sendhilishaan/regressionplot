import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from typing import TextIO
from Models.linear import LinearRegression

SAMPLE_TEST_PATH2 = 'sample/sample2.csv'
SAMPLE_TEST_PATH3 = 'sample/sample3.csv'


if __name__ == '__main__':
    x = LinearRegression(SAMPLE_TEST_PATH3)
    x.train_with_animation(0.01, 100, 1)
