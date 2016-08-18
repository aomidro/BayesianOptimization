# coding=utf-8
"""
created on 2016/07/28
"""

import numpy as np
from BayesianOptimizer import BayesianOptimizer
from InputSpace import InputSpace


def my_function(x):
    """
    quadratic function

    :param x: value(np.array)
    :return: -x^2
    """
    # return - np.power(x[0], 2.0) - np.power(x[1] + 1.0, 2)
    # return -np.power(x[0], 2.0) * np.exp(-x[0] * x[0] * 0.01)
    return x[0] * np.sin(x[0] * x[0])


def main():
    """
    main
    """

    # parameter_range_list = [list(np.linspace(-2.0, 10.0, 100)),
    #                         list(np.linspace(-10.0, 2.0, 100)), ]
    parameter_range_list = [list(np.linspace(-10.0, 10.0, 100000)), ]

    input_space = InputSpace(input_space_list=parameter_range_list)
    optimizer = BayesianOptimizer(input_space=input_space,
                                  black_box_function=my_function,
                                  max_iteration=32,
                                  coarse_graining_parameter=0.001)
    optimizer.execute_optimization()
    optimizer.plot_posterior_distribution()


if __name__ == '__main__':
    main()
