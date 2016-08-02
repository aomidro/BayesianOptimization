# coding=utf-8
"""
created on 2016/07/28
@author = Takashi TAKAHASHI
"""

import numpy as np
from BayesianOptimizer import BayesianOptimizer
from BayesianOptimizerParameter import BayesianOptimizerParameter
from util import My_xrange


def my_function(x):
    """
    quadratic function

    :param x: value
    :return: -x^2
    """
    return (x[0] * np.sin(x[0]*10.0))


def main():
    """
    main
    """

    bayesian_optimizer_parameter = BayesianOptimizerParameter()

    ''' bo parameter '''

    parameter_range_list = [(-1.0, 1.0, 0.001)]

    bayesian_optimizer = BayesianOptimizer(black_box_function=my_function,
                                           bayesian_optimizer_parameter=bayesian_optimizer_parameter,
                                           parameter_range_list=parameter_range_list,
                                           initial_parameter=np.array([1.2])
                                           )
    ''' optimizer '''

    bayesian_optimizer.execute_optimization()


if __name__ == '__main__':
    main()
