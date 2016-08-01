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
    return -1.0 * (x[0] - 1.2) * (x[0] - 1.2) - (x[1] - 2.0) * (x[1] - 2.0) - (x[2] - 1.45) * (x[2] - 1.45)


def main():
    """
    main
    """

    bayesian_optimizer_parameter = BayesianOptimizerParameter()

    ''' bo parameter '''

    parmeter_range_list = [(0, 2.5, 0.025), (0, 2.5, 0.025), (0, 2.5, 0.025)]

    bayesian_optimizer = BayesianOptimizer(black_box_function=my_function,
                                           bayesian_optimizer_parameter=bayesian_optimizer_parameter,
                                           parameter_range_list=parmeter_range_list,
                                           initial_parameter=np.array([0.5, 1.2412, 1.3])
                                           )
    ''' optimizer '''

    bayesian_optimizer.execute_optimization()


if __name__ == '__main__':
    main()
