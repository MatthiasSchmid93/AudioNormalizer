import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


def plot_find_transient_threshold(find_transient_threshold: Callable, *args):

    threshold = find_transient_threshold(*args)
    signal_array = args[0]

    x_values = np.arange(len(signal_array))

    plt.figure()
    plt.plot(x_values, signal_array.ravel()) 
    plt.title('Plot of the Array') 
    plt.xlabel('Time [s]') 
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    return threshold


def plot_signal_array(signal_array):

    x_values = np.arange(len(signal_array))

    plt.figure()
    plt.plot(x_values, signal_array.ravel()) 
    plt.title('Plot of the Array') 
    plt.xlabel('Time [s]') 
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()