import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

plt.plot(x, y, 'o', label="line1")
plt.plot(x, y, label="Fit")
plt.ylabel("YYYYYY")
plt.show()

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



