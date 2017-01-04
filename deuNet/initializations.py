""" Initialization Functions for deuceNet.

Support:
    he_normal: Gaussian initialization scaled by fan_in
        reference:K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” arXiv:1502.01852 [cs], Feb. 2015.

    glorot_normal: Gaussian initialization scaled by fan_in + fan_out
        reference:X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in International conference on artificial intelligence and statistics, 2010, pp. 249–256.
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
