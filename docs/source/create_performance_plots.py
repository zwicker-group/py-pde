#!/usr/bin/env python3
'''
Code for creating performance plots

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import os
os.environ['NUMBA_NUM_THREADS'] = '1'  # check single thread performance

import functools
import timeit

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pde import UnitGrid
from pde.tools.misc import display_progress, estimate_computation_speed

try:
    import cv2
except ImportError:
    opencv_laplace = None
else:
    opencv_laplace = functools.partial(cv2.Laplacian, ddepth=cv2.CV_64F,
                                       borderType=cv2.BORDER_REFLECT)



def time_function(func, arg, repeat=3):
    """ estimates the computation speed of a function
    
    Args:
        func (callable): The function to test
        arg: The single argument on which the function will be estimate
        repeat (int): How often the function is tested
        
    Returns:
        float: Estimated duration of calling the function a single time
    """
    number = int(estimate_computation_speed(func, arg))
    func = functools.partial(func, arg)
    return min(timeit.repeat(func, number=number, repeat=repeat)) / number
    


def get_performance_data(periodic=False):
    """ obtain the data used in the performance plot
    
    Args:
        periodic (bool): The boundary conditions of the underlying grid
        
    Returns:
        dict: The durations of calculating the Laplacian on different grids
        using different methods
    """
    sizes = 2 ** np.arange(3, 13)

    statistics = {}
    for size in display_progress(sizes):
        data = {}
        grid = UnitGrid([size] * 2, periodic=periodic)
        test_data = np.random.randn(*grid.shape)
        
        for method in ['numba', 'scipy']:
            op = grid.get_operator('laplace', bc='natural', method=method)
            data[method] = time_function(op, test_data)
            
        if opencv_laplace:
            data['opencv'] = time_function(opencv_laplace, test_data)
            
        statistics[int(size)] = data

    return statistics



def plot_performance(performance_data, title=None):
    """ plot the performance data
    
    Args:
        performance_data: The data obtained from calling
            :func:`get_performance_data`.
        title (str): The title of the plot
    """
    plt.figure(figsize=[4, 3])
    
    METHOD_LABELS = {'numba': 'py-pde'}
    
    sizes = np.array(sorted(performance_data.keys()))
    grid_sizes = sizes ** 2
    methods = sorted(performance_data[sizes[0]].keys(), reverse=True)
    
    for method in methods:
        data = np.array([performance_data[size][method] for size in sizes])
        plt.loglog(grid_sizes, data, '.-',
                   label=METHOD_LABELS.get(method, method))
        
    plt.xlim(grid_sizes[0], grid_sizes[-1])
    plt.xlabel('Number of grid points')
    plt.ylabel('Runtime [ms]')
    plt.legend(loc='best')
    
    # fix ticks of y-axis
    locmaj = mpl.ticker.LogLocator(base=10, numticks=12) 
    plt.gca().xaxis.set_major_locator(locmaj)
    
    if title:
        plt.title(title)
        
    plt.tight_layout()



def main():
    """ run main scripts """
    data = get_performance_data(periodic=False)
    plot_performance(data, title="2D Laplacian (reflecting BCs)")
    plt.savefig('performance_noflux.pdf', transparent=True)
    plt.savefig('performance_noflux.png', transparent=True, dpi=200)
    plt.close()
    
    data = get_performance_data(periodic=True)
    plot_performance(data, title="2D Laplacian (periodic BCs)")
    plt.savefig('performance_periodic.pdf', transparent=True)
    plt.savefig('performance_periodic.png', transparent=True, dpi=200)
    plt.close()
    


if __name__ == "__main__":
    main()
    