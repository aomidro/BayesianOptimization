# coding=utf-8
"""
created on 2016/07/28
@author = Takashi TAKAHASHI
"""

import numpy as np


class BayesianOptimizerParameter(object):
    """
    description
    """

    def __init__(self, delta=np.power(10.0, -12.0), beta=4.0, sigma=10.0, coarse_graining=0.85, epoch=4096):
        """

        :param delta: used in MI algorithm (see eq 5.2 in "Gaussian Process Optimization with Mutual Information")
        :param beta: used in UCB algorithm
        :param sigma: measurement noise
        :param coarse_graining: coarse graining parameter (coarse graining in the input space)
        :param epoch: maximum iteration number
        """

        self.delta = delta
        ''' delta '''

        self.beta = beta
        ''' beta '''

        self.sigma = sigma
        ''' sigma '''

        self.epoch = epoch
        ''' epoch '''

        self.coarse_graining = coarse_graining
