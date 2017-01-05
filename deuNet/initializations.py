""" Initialization Functions for deuceNet.

Support:
    he_normal: Gaussian initialization scaled by fan_in
        reference: https://arxiv.org/abs/1502.01852
    glorot_normal: Gaussian initialization scaled by fan_in + fan_out
        reference: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.207.2059
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def he_normal(input_size, seed=2014):
    stddev = 1 / math.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev,seed=seed)

def glorot_normal(input_size,output_size, seed=2014):
    stddev = 1 / math.sqrt(input_size + output_size)
    return tf.truncated_normal_initializer(stddev=stddev,seed=seed)
