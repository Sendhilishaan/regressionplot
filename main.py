import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from typing import TextIO

def test():
    x = [1, 2, 3]
    y = [1, 2, 3]

    plt.plot(x, y, 'o', label="line1")
    plt.plot(x, y, label="Fit")
    plt.ylabel("YYYYYY")
    plt.show()

SAMPLE_TEST_PATH2 = 'sample/sample2.csv'
SAMPLE_TEST_PATH3 = 'sample/sample3.csv'

"""
Goal:
    plot the numpy array as dots
    perform gradient descent
    update the line dynamically using as gradient descent iterates

How to dynamically update?

We can define gradient descent as an iteration
Use this to update our plot using FuncAnimation
    - We would set_data(X, y_pred) for y_pred being our updated linreg

We can show the gradient descent by taking contours? and showing a dot move?

TODO:

implement our gradient descent.

"""
class Regression:
    """
    abstract class for a regression model 
    """
    def __init__(self, file: str) -> None:
        self._file = file

    def read_csv(self):
        raise NotImplementedError

    def gradient_descent(self, x: int, y: int) -> int:
        """one iteration of gradient descent"""
        raise NotImplementedError

    def cost_function(self, x: int, y: int) -> int:
        """MSE cost function"""
        raise NotImplementedError

    def display_graph(self):
        raise NotImplementedError

class LinearRegression(Regression):
    """
    Linear regression class (can we normalize to 0,1)?
    """
    
    def __init__(self, file: str) -> None:
        super().__init__(file)

    def read_csv(self) -> tuple:
        """Returning a tuple (x, y) of numpy arrays"""
        df = pd.read_csv(self._file)
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        return (x, y)
    
    def display_graph(self):
        """Display the data from csv as points on the plot"""
        x, y = self.read_csv()
        for i in range(len(x)):
            plt.plot(x[i], y[i], 'o')
        
        plt.ylabel("Y")
        plt.show()

if __name__ == '__main__':
    x = LinearRegression(SAMPLE_TEST_PATH3)
    x.display_graph()



