# coding=utf-8
"""
created on 2016/08/18
"""
import numpy as np
import kernels
import csv
import acquisitionFunctions
import matplotlib.pyplot as plt


class BayesianOptimizer(object):
    """
    Bayesian Optimizer class
    """

    def __init__(self, input_space, black_box_function, max_iteration=10, measurement_noise=0.01,
                 coarse_graining_parameter=0.9):
        """

        :param input_space: input space (InputSpace)
        :param black_box_function: black box function
        :param max_iteration: max iteration number
        :param measurement_noise: measurement noise
        """
        self.__coarse_graining_parameter = coarse_graining_parameter
        ''' coarse graining parameter '''

        self.iteration = 0
        ''' number of times step '''

        self.black_box_function = black_box_function
        ''' black box function'''

        self.max_iteration = max_iteration
        ''' max iteration '''

        self.input_space = input_space
        ''' input_space '''

        self.__measured_point_in_normalized_input_space = []
        ''' measured point list in normalized input space '''

        self.measured_point_in_raw_input_space = []
        ''' measured point list in raw input space '''

        self.measured_value = []
        ''' measured value list '''

        self.optimal_point = None
        ''' optimal point '''

        self.optimal_value = -999999999
        ''' optimal value '''

        self.__kernel = kernels.MaternKernel()
        ''' kernel '''

        self.__kernel_matrix = None
        ''' kernel matrix '''

        self.__kernel_matrix_tilde = None
        ''' inv(Kernel_matrix - \simga^2 * I) '''

        self.__measurement_noise = measurement_noise
        ''' measurement noise '''

        self.__alternate_function = acquisitionFunctions.MutualInformation()
        # self.__alternate_function = alternateFunctions.UpperConfidenceBound()
        ''' alternate_function '''

        self.__initialize()

    def __initialize(self):
        """
        initialization

        :return:
        """
        self.measured_point_in_raw_input_space.append(np.array([x[0] for x in self.input_space.raw_input_space_list]))
        self.__measured_point_in_normalized_input_space.append(
            np.array([x[0] for x in self.input_space.normalized_input_space_list]))
        measured_value = self.black_box_function([x[0] for x in self.input_space.raw_input_space_list])
        self.measured_value.append(measured_value)
        self.__kernel_matrix = \
            np.array([[self.__kernel.compute_value(self.__measured_point_in_normalized_input_space[0],
                                                   self.__measured_point_in_normalized_input_space[0])]])
        self.__update_optimum(measured_value=measured_value,
                              next_raw_measurement_point=np.array(
                                  [x[0] for x in self.input_space.raw_input_space_list]))
        print(self.__kernel_matrix)
        self.print_on()

    def execute_optimization(self):
        """
        execute optimization
        :return: optimal_point, optimal_value
        """
        while self.iteration < self.max_iteration:
            self.__step()
            self.iteration += 1
        return self.optimal_point, self.optimal_value

    def __step(self):
        """
        execute exploration one time
        :return: None
        """
        # get kernel matrix tilde
        self.__update_kernel_matrix_tilde()

        # find next measurement point in the input space
        next_raw_measurement_point, next_normalized_measurement_point, mean_at_next_measurement_point, variance_at_next_measurement_point \
            = self.__find_next_measurement_point()

        # do measurement
        measured_value = self.__do_measurement(next_raw_measurement_point, next_normalized_measurement_point)

        # update optimal value
        self.__update_optimum(measured_value, next_raw_measurement_point)

        # update kernel matrix
        self.__update_kernel_matrix(next_normalized_parameter_set=next_normalized_measurement_point)

        # update alternate function state
        self.__alternate_function.update_state(mean_at_next_measurement_point, variance_at_next_measurement_point)

        # print out current state
        self.print_on()

    def __find_next_measurement_point(self):
        """
        find next measurement point based on the estimated value of the alternate function
        :return: next measurement point (in raw input space)
        """

        # find optimal next point in the input space
        optimal_alternate_function_value = -99999999
        raw_candidate_point = None
        normalized_candidate_point = None
        mean_at_next_measurement_point = None
        variance_at_next_measurement_point = None

        while raw_candidate_point is None:
            # get input space
            raw_input_space, normalized_input_space = self.input_space.get_input_space()
            for raw_point in raw_input_space:
                raw_point = np.array(raw_point)
                normalized_point = np.array(normalized_input_space.next())

                if np.random.rand() < np.power(self.__coarse_graining_parameter, self.input_space.dimension):
                    # get mean and variance at normalized measurement point
                    mean, variance = self.__get_mean_variance(normalized_point)

                    # get alternate function value
                    candidate_value = self.__alternate_function.get_value(mean, variance)
                    if np.round(optimal_alternate_function_value, 7) < np.round(candidate_value, 7):
                        optimal_alternate_function_value = candidate_value
                        normalized_candidate_point = normalized_point
                        raw_candidate_point = raw_point
                        mean_at_next_measurement_point = mean
                        variance_at_next_measurement_point = variance

        return raw_candidate_point, normalized_candidate_point, mean_at_next_measurement_point, variance_at_next_measurement_point

    def __get_mean_variance(self, normalized_candidate_measurement_point):
        """
        get mean and variance at measurement point
        :return: mean, variance
        """
        x = np.array(normalized_candidate_measurement_point)
        # calc k, \hat{k}
        k_T = np.array([self.__kernel.compute_value(x, x_prime)
                        for x_prime in self.__measured_point_in_normalized_input_space])
        k_T_hat = self.__kernel.compute_value(x, x) - np.dot(np.dot(k_T, self.__kernel_matrix_tilde), k_T)

        # calc mean
        mean = np.dot(np.dot(k_T, self.__kernel_matrix_tilde), np.array(self.measured_value))

        # calc variance
        variance = k_T_hat

        return mean, variance

    def __do_measurement(self, next_raw_measurement_point, next_normalized_measurement_point):
        """
        do measurement of the black box function
        :param next_raw_measurement_point:
        :param next_normalized_measurement_point:
        :return:
        """
        # add measurement point
        self.measured_point_in_raw_input_space.append(next_raw_measurement_point)
        self.__measured_point_in_normalized_input_space.append(next_normalized_measurement_point)

        # get black box function value
        measured_value = self.black_box_function(next_raw_measurement_point) + \
                         np.random.normal(loc=0.0, scale=self.__measurement_noise)

        # add measured value
        self.measured_value.append(measured_value)

        return measured_value

    def __update_kernel_matrix(self, next_normalized_parameter_set):
        """
        update kernel matrix
        :param next_normalized_parameter_set:
            next point in normalized input space kernel matrix will be update based on this vector
        :return: None
        """
        pre_append_vector = [np.array([self.__kernel.compute_value(x, next_normalized_parameter_set)])
                             for x in self.__measured_point_in_normalized_input_space]
        self.__kernel_matrix = np.append(self.__kernel_matrix, np.array(pre_append_vector[:-1]), axis=1)
        self.__kernel_matrix = np.append(self.__kernel_matrix,
                                         np.array([np.array(
                                             [self.__kernel.compute_value(next_normalized_parameter_set, x) for x in
                                              self.__measured_point_in_normalized_input_space])]),
                                         axis=0)

    def __update_kernel_matrix_tilde(self):
        """
        update kernel matrix tilde
        :return: None
        """
        self.__kernel_matrix_tilde = np.linalg.inv(
            self.__kernel_matrix +
            np.power(self.__measurement_noise, 2.0) * np.identity(self.__kernel_matrix.shape[0])
        )

    def __update_optimum(self, measured_value, next_raw_measurement_point):
        """
        update optimum
        :param measured_value:
        :param next_raw_measurement_point:
        :return: None
        """
        if self.optimal_value < measured_value:
            self.optimal_value = measured_value
            self.optimal_point = next_raw_measurement_point

    def print_on(self):
        """
        print current state
        """
        print("##### current optimizer status #####")
        print("current iteration step -> " + str(self.iteration))
        print("measured points -> ")
        print("    " + str(self.__measured_point_in_normalized_input_space))
        print("measured values -> ")
        print("    " + str(self.measured_value))
        print("current optimal point ->" + str(self.optimal_point))
        print("current optimal value ->" + str(self.optimal_value))
        print("")

    def write_exploration_history(self, file_path=None):
        """
        write out exploration history
        :param file_path: output file path
        """
        if file_path is None:
            file_path = "result_at_iteration" + str(self.iteration) + ".csv"
        writer = csv.writer(open(file_path, "w"))
        measurement_result = zip(self.measured_value, self.__measured_point_in_normalized_input_space)
        for result in measurement_result:
            writer.writerow(list(result))

    def plot_posterior_distribution(self, file_path=None):
        """
        plot posterior distribution
        this method works only if the input space dimension is one
        :param file_path: output file path
        :return: None
        """
        if len(self.input_space.raw_input_space_list) != 1:
            print("expected input space dimension is one ")
            print("input space dimension is " + str(len(self.input_space.raw_input_space_list)))
            print("")
        else:
            self.__update_kernel_matrix_tilde()

            mean_var = [self.__get_mean_variance(np.array([x])) for x
                        in self.input_space.normalized_input_space_list[0]]
            mean = [x[0] for x in mean_var]
            stddev = np.sqrt([x[1] for x in mean_var])

            higher_bound = mean + stddev
            lower_bound = mean - stddev
            x = self.input_space.raw_input_space_list[0]

            plt.fill_between(x, higher_bound, lower_bound, color="blue", label="confidence")
            plt.plot(x, mean, color="red", label="estimated")

            plt.legend()
            plt.grid()
            plt.show()
