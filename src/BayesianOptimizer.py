# coding=utf-8
"""
created on 2016/07/28
@author = Takashi TAKAHASHI
"""

import numpy as np
import kernels
import itertools
import time
from util import My_xrange


class BayesianOptimizer(object):
    """
    Bayesian Optimizer class
    Algorithm is based on the GP-MI(see "Gaussian Process Optimization with Mutual Information").

    ()
    """

    def __init__(self, black_box_function, bayesian_optimizer_parameter, parameter_range_list, initial_parameter):
        """
        constructor
        :param bayesian_optimizer_parameter: bayesian_optimizer parameter
        :param black_box_function: black-box function
        :param parameter_range_list: parameter range list [(min1, max1, pitch1), (min2, max2, pitch2), ...]
        """

        self.black_box_function = black_box_function
        ''' black box function '''

        self.bayesian_optimizer_parameter = bayesian_optimizer_parameter
        ''' bayesian optimizer parameter '''

        self.parameter_range_list = parameter_range_list
        ''' parameter range list '''

        # self.kernel = kernels.GaussianKernel()
        self.kernel = kernels.MaternKernel()
        ''' kernel '''

        self.kernel_matrix = np.array([[self.kernel.compute_value(initial_parameter, initial_parameter)]])
        ''' kernel matrix '''

        self.kernel_matrix_tilde = None
        ''' inv(Kernel_matrix - \simga^2 * I)'''

        self.gamma = 0.0
        ''' gamma '''

        self.measured_point = [initial_parameter]
        ''' measured point '''

        self.measured_value = [self.black_box_function(initial_parameter)]
        ''' measured value '''

        self.optimal_value = -99999999999.0
        ''' optimal value of the evaluation function '''

        self.optimal_parameter = [0.0] * len(parameter_range_list)
        ''' optimal parameter set '''

        self.input_space = [My_xrange(x[0], x[1], x[2]) for x in parameter_range_list]
        ''' input space (iterator list) '''

    def execute_optimization(self):
        """
        execute optimization method

        :return: optimized value
        """

        # initialization
        counter = 0
        optimal_value_of_black_box_function = -99999999999999999.0
        optimal_parameter_of_black_box_function = None
        while counter < self.bayesian_optimizer_parameter.epoch:
            counter += 1
            # inv_matrix
            self.__update_kernel_matrix_tilde()

            # get next parameter: x_candidate
            x_candidate = None
            optimal_evaluation_function_value = -99999999999.0
            variance_at_x_candidate = 0.0

            for parameter in itertools.product(*self.input_space):
                if np.random.rand() < np.power(self.bayesian_optimizer_parameter.coarse_graining,
                                               len(self.parameter_range_list)):
                    x = np.array(list(parameter))
                    # calc k, \hat{k}
                    k_T = np.array([self.kernel.compute_value(x, x_prime) for x_prime in self.measured_point])
                    k_T_hat = self.kernel.compute_value(x, x) - np.dot(
                        np.dot(k_T, self.kernel_matrix_tilde),
                        k_T)

                    # calc mean
                    mean = np.dot(np.dot(k_T, self.kernel_matrix_tilde), np.array(self.measured_value))
                    # calc variance
                    var = k_T_hat

                    # update x_candidate
                    # UCB
                    # optimal_evaluation_function_value, x_candidate = self.__update_parameter_UCB(
                    #     optimal_evaluation_function_value=optimal_evaluation_function_value, mean=mean, var=var, x=x,
                    #     x_candidate=x_candidate)
                    # MI
                    optimal_evaluation_function_value, x_candidate, variance_at_x_candidate = self.__update_parameter_MI(
                        optimal_evaluation_function_value=optimal_evaluation_function_value, mean=mean, var=var, x=x,
                        x_candidate=x_candidate)

            self.gamma = self.gamma + variance_at_x_candidate
            next_parameter_set = np.round(x_candidate, 7)
            print("iteration: " + str(counter))
            print(" next_value -> " + str(next_parameter_set))
            print(" current_best -> " + str(self.optimal_parameter))
            print("")

            # do measurement
            self.measured_point.append(np.array(next_parameter_set))
            y = self.black_box_function(next_parameter_set) + np.random.normal(loc=0,
                                                                               scale=self.bayesian_optimizer_parameter.sigma)
            if self.optimal_value < y:
                self.optimal_value = y
                self.optimal_parameter = next_parameter_set

            self.measured_value.append(y)

            # update kernel matrix
            pre_append_vector = [np.array([self.kernel.compute_value(x, next_parameter_set)]) for x in
                                 self.measured_point]
            # pre_append_vector.pop()

            self.kernel_matrix = np.append(self.kernel_matrix, np.array(pre_append_vector[:-1]), axis=1)
            self.kernel_matrix = np.append(self.kernel_matrix, np.array(
                [np.array([self.kernel.compute_value(next_parameter_set, x) for x in self.measured_point])]), axis=0)

        return optimal_parameter_of_black_box_function, optimal_value_of_black_box_function

    def __update_kernel_matrix_tilde(self):
        """
        caompute kernel matrix tilde
        :return: None
        """

        self.kernel_matrix_tilde = np.linalg.inv(
            self.kernel_matrix + self.bayesian_optimizer_parameter.sigma * self.bayesian_optimizer_parameter.sigma * np.identity(
                self.kernel_matrix.shape[0]))

    def __update_parameter_UCB(self, optimal_evaluation_function_value, mean, var, x, x_candidate):
        """

        :param optimal_evaluation_function_value:
        :param mean: mean of the posterior
        :param var: variance of the posterior
        :param x: parameter candidate
        :param x_candidate: next parameter candidate
        :return: optimal_optimal_evaluation_function_value, next parameter candidate
        """

        if optimal_evaluation_function_value + 0.001 < (mean + np.sqrt(self.bayesian_optimizer_parameter.beta * var)):
            return (mean + np.sqrt(self.bayesian_optimizer_parameter.beta * var)), list(x)
        else:
            return optimal_evaluation_function_value, x_candidate

    def __update_parameter_MI(self, optimal_evaluation_function_value, mean, var, x, x_candidate):
        """

        :param optimal_evaluation_function_value:
        :param mean:
        :param var:
        :param x:
        :param x_candidate:
        :return:
        """
        alpha = np.log(2.0 / self.bayesian_optimizer_parameter.delta)
        evaluation_value = mean + np.sqrt(alpha) * var / (np.sqrt(var + self.gamma) + np.sqrt(self.gamma))
        if np.round(optimal_evaluation_function_value, 7) < np.round(evaluation_value, 7):
            return evaluation_value, list(x), var
        else:
            return optimal_evaluation_function_value, x_candidate, 0.0
