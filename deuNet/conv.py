"""Immplementation of convolutional nn modules

Classes defining convolutional operations, derived from `deuNet.AbstractModule`, with easy weight sharing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numbers

import numpy as np
import tensorflow as tf

from deuNet import base
from deuNet import util
from deuNet import initializations

from IPython import embed

SAME = "SAME"
VALID = "VALID"
ALLOWED_PADDINGS = {SAME, VALID}

def _fill_shape(x,n):
    """Automatically create a tuple of shape repeating x by n times.
    
    Inputs:
        x: int or tuple of ints.
        n: int. number of tuples we want to repeat.
    Returns:
        if `x` is int, a tuple of size `n` containing `n` copies of `x`
        if `x` is tuple, return itself
        if `x` is list, return tuple
    For example, x = 10, n = 2, output = (10, 10)
    """
    if not isinstance(n, numbers.Integral) or n < 1:
        raise TypeError("n must be a positive integer.")

    if isinstance(x, numbers.Integral):
        return (x, ) * n
    elif isinstance(x,tuple):
        return x
    elif isinstance(x,list):
        return tuple(x)
    else:
        raise TypeError("x is {}, must be either int or tuple or list.".format(x))

def _verify_padding(padding):
    """Verify padding method"""
    if padding not in ALLOWED_PADDINGS:
        raise ValueError("Padding must be member of {}, but not {}".format(ALLOWED_PADDINGS,padding))
    return padding

class Conv2D(base.AbstractModule):
    """Spatial convolution convolutional module, including bias
    
    This acts as an easy warpper around the `tf.nn.conv2d` and `tf.nn.atrous_conv2d`, abstracting away variable creation and sharing.
    It accepts tensorflow dim order for images: <batch, height, width, channels>
    """

    def __init__(self, output_channels, kernel_shape, stride=1,
            padding=SAME,initial_params=None, activation=None, use_bias=True, initializer=None, name="conv_2d"):
        """Conv2D constructor
        
        tensorflow has an explaination of VALID and SAME padding modes:
        https://www.tensorflow.org/api_docs/python/nn/convolution#convolution

        Inputs:
            output_channels: int, number of ouptut channels. 
            kernel_shape: tuple/list of int, defines the kernel sizes.
            stride: tuple of int to define strides in height and width, or an interger that is used to define stride in all dims.
            padding: padding method, either `SAME` or `VALID`
            initial_params: dict of np.array, defining initial parameters of weights, filter shape = <filter_height, filter_width, in_channels, out_channels>
            use_bias: bool, whether use bias
            initializer: string, define which initialize method to use
            name: string, name of the module

        Given a 2-D convolution given 4-D `input` tensor and `filter` variable, it performs local region transformation in a sliding-window fasion (sliding window method is defined by stride and padding method)

        Given an input tensor of shape <batch, in_height, in_width, in_channels> and a filter/kernel variable of shape <filter_height, filter_width, in_channels, out_channels>, this module performs the following:

         - 1. Flatten the filter `tf.variable` to a 2-D matrix with shape <filter_height * filter_width * in_channels, output_channels>
         - 2. Extracts image patches (sliding window) from the input tensor to form a virtual tensor of shape <batch, out_height, out_width, filter_height * filter_width * in_channels>. `out_height` and `out_width` is computed by padding method and stride.
         - 3. For each patch, right-multiplies the filter matrix and the image patch vector. Finally, it produces the output of shape <batch, out_height, out_width, output_channels>
        """
        super(Conv2D, self).__init__(name=name)

        self._output_channels = output_channels
        self._kernel_shape = _fill_shape(kernel_shape,2)
        # create stride, tensorflow stride is a 4-D array, stride of <batch, height, width, channels>. Normally, we only move in height and width but not batch and channels.
        try:
            self._stride = (1,) + _fill_shape(stride,2) + (1,)
        except TypeError as e:
            if len(stride) == 4:
                self._stride = tuple(stride)
            else:
                raise base.IncompatibleShapeError("Invalid stride: {}".format(e))

        self._padding = _verify_padding(padding)
        self._use_bias = use_bias
        self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
        self._initial_params = initial_params
        self.initializer = initializer
        self._activation = util.check_activation(activation)

    @classmethod
    def get_possible_initializer_keys(cls, use_bias=True):
        return {'w','b'} if use_bias else {'w'}


    def _build(self, inputs):
        """Connects the Conv2D module into the graph, with input Tensor `inputs`
        
        We create `tf.variable` for module parameters and call the `tf.nn.conv2d` function here.

        Inputs:
            inputs: A 4D tensor of shape <batch_size, input_height, input_width, input_channels>
        Returns:
            outputs: tf.tensor, shape = <batch_size, output_height, output_width, output_channels>
        """
        self._input_shape = tuple(inputs.get_shape().as_list())
        input_height = self._input_shape[1]
        input_width = self._input_shape[2]
        input_channels = self._input_shape[3]
        if self.padding == SAME:
            out_height = np.ceil(float(input_height) / float(self._stride[1]))
            out_width = np.ceil(float(input_width) / float(self._stride[2]))
        elif self.padding == VALID:
            out_height = np.ceil(float(input_height - kernel_shape[0] + 1) / float(self._stride[1]))
            out_width = np.ceil(float(input_width - kernel_shape[1] + 1) / float(self._stride[2]))
        self._output_shape = (self._input_shape[0], out_height, out_width, self.output_channels)

        if len(self._input_shape) != 4:
            raise base.IncompatibleShapeError("Input Tensor must have shape (batch_size, in_height, in_width, in_channels)")
        
        # handle `tf.variable` shapes
        kernel_shape = self.kernel_shape
        param_shapes = {}
        dtype = inputs.dtype
        weight_shape = (kernel_shape[0], kernel_shape[1], input_channels, self.output_channels)
        param_shapes["w"] = weight_shape
        if self._use_bias:
            bias_shape = (self.output_channels,) 
            param_shapes["b"] = bias_shape
        
        self._initializers = util.get_initializers(self.possible_keys, self.initializer, param_shapes, init_params=self._initial_params,conv=True)

        self._w = util.get_tf_variable("w", shape=weight_shape, dtype=dtype, initializer=self._initializers["w"])
        outputs = tf.nn.conv2d(inputs, self._w, strides = self.stride, padding = self.padding)
        
        if self._use_bias:
            self._b = util.get_tf_variable("b", shape=bias_shape, dtype=dtype, initializer=self._initializers["b"])
            outputs += self._b

        if self.activation is not None:
            outputs = self.activation(outputs)

        if tuple(outputs.get_shape().as_list()) != self._output_shape:
            raise base.IncompatibleShapeError("real output shape {} must agree with theoretical shape {}".format(
                tuple(outputs.get_shape().as_list()),self._output_shape))
        return outputs

    @property
    def output_channels(self):
        return self._output_channels
    @property
    def kernel_shape(self):
        return self._kernel_shape
    @property
    def padding(self):
        return self._padding
    @property
    def stride(self):
        return self._stride
    @property
    def w(self):
        self._ensure_is_connected()
        return self._w
    @property
    def b(self):
        self._ensure_is_connected()
        return self._b
    @property
    def input_shape(self):
        self._ensure_is_connected()
        return self._input_shape
    @property
    def output_shape(self):
        self._ensure_is_connected()
        return self._output_shape
    @property
    def activation(self):
        return self._activation


class MaxPool(base.AbstractModule):
    """Spatial max-pooling module"""
    def __init__(self, kernel_shape, stride=1, padding=SAME, name="max_pool"):
        super(MaxPool,self).__init__(name=name)

        try:
            self._kernel_shape = (1,) + _fill_shape(kernel_shape,2) + (1,)
        except TypeError as e:
            if len(kernel_shape) == 4:
                self._kernel_shape = kernel_shape
            else:
                raise base.IncompatibleShapeError("Invalid kernel shape: {}".format(e))

        try:
            self._stride = (1,) + _fill_shape(stride,2) + (1,)
        except TypeError as e:
            if len(stride) == 4:
                self._stride = stride
            else:
                raise base.IncompatibleShapeError("Invalid stride: {}".format(e))

        self._padding = _verify_padding(padding)
        
    def _build(self, inputs):
        """Connects the MaxPool module into the graph"""
        self._input_shape = tuple(inputs.get_shape().as_list())
        outputs = tf.nn.max_pool(inputs, ksize=self._kernel_shape, strides=self._stride, padding=self._padding)

        return outputs
