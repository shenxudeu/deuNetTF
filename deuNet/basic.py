"""Basic Modules for deuceNet.

This defines basic building blocks for NN.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers


import numpy as np
import tensorflow as tf

from deuNet import base
from deuNet import util
from deuNet import initialization

class Linear(base.AbstractModule):
    """Linear Module with Bias(optional)"""

    def __init__(self, output_size, use_bias=True, initializer=None, name="linear"):
        super(Linear, self).__init__(name=name)
        self._output_size = output_size
        self._use_bias = use_bias
        self._input_shape = None
        self._w = None
        self._b = None
        self.possible_keys = self.get_possbile_initializer_keys(use_bias=use_bias)
        ''' TOBE IMPLEMENTED in util
        self._initializers = util.check_initializers(initializers, self.possible_keys)
        '''


       


