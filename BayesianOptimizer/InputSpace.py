# coding=utf-8

import itertools


class InputSpace(object):
    """
    Note:
        Input Space class

    Args:
        input_space_list: input space of each dimensions

    Attriibutes:
        raw_input_space_list: raw input space
        normalized_input_space_list : input space normalized onto hyper unit cube
    """

    def __init__(self, input_space_list):
        self.raw_input_space_list = [sorted(x) for x in input_space_list]
        ''' raw input space list '''
        self.normalized_input_space_list = [[(x - parameter_list[0]) / (parameter_list[-1] - parameter_list[0])
                                             for x in parameter_list] for parameter_list in self.raw_input_space_list]
        ''' normalized input space list '''

        self.dimension = len(self.raw_input_space_list)
        ''' input space dimension'''

        # self.print_on()

    def get_input_space(self):
        """
        get input space(iter)
        :return: (raw_input_space, normalized_input_space)
        """
        return itertools.product(*self.raw_input_space_list), itertools.product(*self.normalized_input_space_list)

    def print_on(self):
        print("##### raw input space #####")
        for x in self.raw_input_space_list:
            print(x)
        print("")
        print("##### normalized input space #####")
        for x in self.normalized_input_space_list:
            print(x)
        print("")
        print("##### dimension #####")
        print(self.dimension)
        print("")
