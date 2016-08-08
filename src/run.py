# coding=utf-8
"""
created on 2016/07/28
"""

import numpy as np
from util import MyXrange
from BayesianOptimizer import BayesianOptimizer
from BayesianOptimizerParameter import BayesianOptimizerParameter


def my_function(x):
    """
    quadratic function

    :param x: value
    :return: -x^2
    """
    return -x[0] * x[0]


def main():
    """
    main
    """

    bayesian_optimizer_parameter = BayesianOptimizerParameter()
    ''' bo parameter '''

    # parameter_range_list = [My_xrange(0, 2.5, 0.1)]
    parameter_range_list = [np.linspace(-2.5, 2.5, 1000)]

    bayesian_optimizer = BayesianOptimizer(black_box_function=my_function,
                                           bayesian_optimizer_parameter=bayesian_optimizer_parameter,
                                           parameter_range_list=parameter_range_list,
                                           initial_parameter=np.array([1.2])
                                           )
    ''' optimizer '''

    a,b = bayesian_optimizer.execute_optimization()
    ''' execute optimization '''

    print(a)
    print(b)


if __name__ == '__main__':
    main()
