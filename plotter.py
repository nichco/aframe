import numpy as np
import matplotlib.pyplot as plt

def plotter(options, U):

    # plot each element:
    for element_name in options:
        node_a = options[element_name]['node_a']
        node_b = options[element_name]['node_b']

