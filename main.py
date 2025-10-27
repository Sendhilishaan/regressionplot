import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from typing import TextIO

SAMPLE_TEST_PATH2 = 'sample/sample2.csv'
SAMPLE_TEST_PATH3 = 'sample/sample3.csv'

class Regression:
    """
    abstract class for a regression model 
    """
    def __init__(self, file: str) -> None:
        self._file = file

    def read_csv(self):
        raise NotImplementedError

    def gradient_descent_step(self) -> int:
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
        plt.xlabel("X")
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

    def gradient_descent_step(self):
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
    
    def train(self, tolerance: float, max_iterations: int):
        """Train the model and update our graph"""
        prev_cost = self.cost_function()

        for i in range(max_iterations):
            
            self.gradient_descent_step()
            cost = self.cost_function()

            if abs(prev_cost - cost) < tolerance:
                # Include print statements here?
                break

            prev_cost = cost
    
    def train_with_animation(self, tolerance: float, max_iterations: int, update_every=10):
        """Train the model with live animation"""
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

        prev_cost = self.cost_function()
        print(f"Initial cost: {prev_cost:.6f}")

        for i in range(max_iterations):
            self.gradient_descent_step()
            cost = self.cost_function()
            
            # Update plot every N iterations
            if i % update_every == 0:
                ax.clear()
                
                # Plot data points
                ax.scatter(self.x, self.y, color='blue', label='Data', s=50, zorder=3)
                
                # Plot current regression line
                y_pred = self.m * self.x + self.b
                ax.plot(self.x, y_pred, color='red', linewidth=2, 
                        label=f'Iter {i}: y = {self.m:.2f}x + {self.b:.2f}')
                
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f'Cost: {cost:.6f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.pause(0.05)  # Pause to update display
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
            
            if abs(prev_cost - cost) < tolerance:
                print(f"Converged at iteration {i}")
                break
            
            prev_cost = cost

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep final plot open

if __name__ == '__main__':
    x = LinearRegression(SAMPLE_TEST_PATH3)
    x.train_with_animation(0.01, 100, 10)
