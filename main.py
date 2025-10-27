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
    Linear regression class (should we normalize to 0,1)?
    """
    
    def __init__(self, file: str) -> None:
        super().__init__(file)
        self.m = 0
        self.b = 0
        self.alpha = 0.01

        # Initialise our x, y
        df = pd.read_csv(self._file)
        self.x = df['x'].to_numpy()
        self.y = df['y'].to_numpy()
        self.n = len(self.x)

    
    def display_graph(self):
        """Display the data from csv as points on the plot"""
        plt.clf()
        plt.scatter(self.x, self.y) 
        plt.ylabel("Y")
        plt.show()

    def cost_function(self) -> float:
        """Mean squared error cost function for linear regression"""
        sum = 0

        for i in range(self.n):
            sum += (self.current_function(self.x[i]) - self.y[i]) ** 2
        
        return sum / self.n
    
    def current_function(self, x: int) -> int:
        """Calculate the output of our current linear function given x"""
        return (self.m * x) + self.b

    def gradient_descent(self, x: int, y: int):
        """
        One iteration of gradient descent for linear regression
        Updates our parameters self.m and self.y
        """
        predictions = self.m * self.x + self.b
        errors = predictions - self.y

        # Gradients
        dm = (2/self.n) * np.sum(errors * self.x)
        db = (2/self.n) * np.sum(errors)

        self.m -= self.alpha * dm
        self.b -= self.alpha * db


if __name__ == '__main__':
    x = LinearRegression(SAMPLE_TEST_PATH3)
    for i in range(5):
        x.display_graph()

    #plt.plot([1, 10], [1, 10])
    #plt.show()