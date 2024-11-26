import numpy as np
import matplotlib.pyplot as plt

def straight_line_function(x, m, c):
    """
    Returns basic straight line function: y = m * x + c
    """
    return m * x + c

def inverse_straight_line_function(y, m, c, entry_x):
    """
    Returns inverse of basic straight line function: x = (y - c) / m
    """
    if abs(m) > 1e-10:
        with np.errstate(divide='ignore', invalid='ignore'):
            return (y - c) / m
    else:
        if type(y) is int:
            return entry_x
        else:
            return np.linspace(entry_x, entry_x, len(y))

def calc_distance(coord_1, coord_2):
    """
    Calculates the distance between two coordinates
    """
    x_1 = coord_1[:,0]
    y_1 = coord_1[:,1]
    x_2 = coord_2[:,0]
    y_2 = coord_2[:,1]

    delta_x = x_2 - x_1
    delta_y = y_2 - y_1

    return np.sqrt(delta_x ** 2 + delta_y ** 2)

