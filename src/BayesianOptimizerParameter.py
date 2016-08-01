# coding=utf-8
"""
created on 2016/07/28
@author = Takashi TAKAHASHI
"""


class BayesianOptimizerParameter(object):
    """
    description
    """

    def __init__(self, beta=1.0, sigma=0.001, coarse_graining=0.2, epoch=100):
        """

        :param beta: relative weight between mean and variance
        :param sigma: noise
        """
        self.beta = beta
        ''' beta '''

        self.sigma = sigma
        ''' sigma '''

        self.epoch = epoch
        ''' epoch '''

        self.coarse_graining = coarse_graining
