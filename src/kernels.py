#  coding=utf-8
"""
created on 2016/07/28
"""

import numpy as np


class GaussianKernel(object):
    """
    GaussianKernel class

    the value is computed as follows
    \alpha_0 * \exp(-\alpha_1* ||x1 - x2||_2^2 / 2.0)

    \alpha_0 * \exp(−(2\alpha_1^2)^{−1}||x1 − x2||^2),
    description
    """

    def __init__(self, alpha0=1.0, alpha1=1.00):
        """

        :param alpha0: parameter \alpha_0
        :param alpha1: parameter \alpha_1
        """

        self.alpha0 = alpha0
        ''' alpha_0 '''
        self.alpha1 = alpha1
        ''' alpha_1 '''

    def compute_value(self, x1, x2):
        """
        compute the value of the gaussian kernel

        :param x1: input vector 1 (np.array)
        :param x2: input vector 2 (np.array)
        :return: the value of the kernel (np.float32)
        """

        return np.float32(self.alpha0 * np.exp(-1.0 * np.dot(x1 - x2, x1 - x2) / 2.0 / self.alpha1 / self.alpha1))


class MaternKernel(object):
    """
    Matern Kernel with nu=5/2 class

    the value is computed as follows

    general expression:
        k(x1, x2) = (2^{1−ν} /Γ(ν))r^ν B_ν(r), r = \sqrt{2ν/l}||x2 − x2||
    the nu = 5/2 case:
        k(x1, x2) = (1+\sqrt(5)*r + (5/3)*r^2 ) * \exp(-\sqrt(5) * r)
        r = d / l
        d = np.abs(x1-x2)
        l : the length scale

    """

    def __init__(self, l=0.1):
        self.l = l

    def compute_value(self, x1, x2):
        """
        compute the value of the Matern Kernel
        :param x1: input vector 1 (np.ndarray)
        :param x2: input vector 2 (np.ndarray)
        :return: the value of the kernel (np.float32)
        """
        r = np.linalg.norm(x1 - x2) / self.l
        return (1.0 + np.sqrt(5) * r + (5.0 / 3.0) * r * r) * np.exp(-1.0 * np.sqrt(5) * r)
