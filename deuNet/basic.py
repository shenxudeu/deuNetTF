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

    def __init__(self, output_size, use_bias=True, initializers=None, name="linear"):
        super(Linear, self).__init__(name=name)
        self._output_size = output_size
        self._use_bias = use_bias
        self._input_shape = None
        self._w = None
        self._b = None
        self.possible_keys = self.get_possbile_initializer_keys(use_bias=use_bias)
        # check if all `tf.Variable` have initializer.
        self._initializers = util.check_initializers(initializers, self.possible_keys)

    @classmethod
    def get_possible_initializer_keys(cls, use_bias=True):
        return {'w','b'} if use_bias else {'w'}

    def _build(self, inputs):
        """Connects the linear module into the graph, which input Tensor `inputs`
        
        Inputs:
            inputs: tf.Tensor, 2-D tensor with size = [batch_size, feature_size]
        Returns:
            outputs: tf.Tensor, 2-D tensor with size = [batch_size, output_size]
        """
        # NOTE: how to get shape of a `tf.Variable`
        # 1. static shape: In graph construction time, we can get static shape of `tf.Variable` by calling tf.Variable.get_shape().as_list()
        # 2. dynamic shape: In runtime, we can get dynamic shape of `tf.Variable` by calling tf.shape(tf.Variable)
        # reference: http://stackoverflow.com/questions/34079787/tensor-with-unspecified-dimension-in-tensorflow
        input_shape = tuple(inputs.get_shape().as_list())
        
        if len(input_shape) != 2:
            raise base.IncompatibleShapeError("{}: rank of shape must be 2 not: {}".format(self.name, len(input_shape)))

        if input_shape[1] is None:
            raise base.IncompatibleShapeError("{}: input_size(feature_size) must be specified at module build time".format(self.name))

        if self._input_shape is not None and input_shape[1] != self._input_shape[1]:
            raise base.IncompatibleShapeError("{}: input shape must be [batch_size, {}] but not: [batch_size, {}]".format(self.name, self._input_shape[1], input_shape[1]))

        self._input_shape = input_shape

        if "w" not in self._initializers:
            self._initializers["w"] = initialization.he_normal(self._input_shape[1])
        if "b" not in self._initializers:
            self._initializers["b"] = initialization.he_normal(self._input_shape[1])

        weight_shape = (self._input_shape[1], self.output_size)
        dtype = inputs.dtype

        # NOTE: use `tf.get_variable` instead of `tf.Variable` for weight sharing
        self._w = tf.get_variable("w", shape=weight_shape, dtype=dtype, initializer=self._initializers["w"])
        outputs = tf.matmul(inputs, self._w)
        
        if self._use_bias:
            bias_shape = (self.output_size,)
            self._b = tf.get_variable("b", shape=bias_shape, dtype=dtype, initializer=self._initializers["b"])
            otuputs = outputs + self._b

        return outputs

    @property
    def w(self):
        self._ensure_is_connected()
        return self._w

    @property
    def b(self):
        self._ensure_is_connected()
        if not self._use_bias:
            raise AttributeError("{}: No bias variable in Linear Module if `use_bias=False`".format(self.name))
        return self._b

    @property
    def output_size(self):
        return self._output_size

    @property
    def input_size(self):
        return self._input_shape


class Dense(Linear):
    """Dense Module with Bias(optional)
    
    This is derived from Linear module, but with a activation function.
    """

    def __init__(self, output_size, activation=None, use_bias=True, initializers=None, name="dense"):
        """ Dense constructor

        Inputs:
            output_size: int, output of this module will be [batch_size, output_size]
            activation: `tf.callable`, element-wise function used as activation function. Default is `None`, which is linear module.
            use_bias: bool, weather to include bias parameters. Default `True`
            initializers: dict, optional dict containing initializers to initialize the weights and bais.(There will be a wrapper to
                generate this dict)
            name: string, name of this module
        """
        super(Dense, self).__init__(name=name)
        self._output_size = output_size
        self._activation = util.check_activation(activation)
        self._use_bias = use_bias
        self._input_shape = None
        self._w = None
        self._b = None
        self.possible_keys = self.get_possbile_initializer_keys(use_bias=use_bias)
        # check if all `tf.Variable` have initializer.
        self._initializers = util.check_initializers(initializers, self.possible_keys)

    
    def _build(self, inputs):
        """Connects the dense module into the graph, which input Tensor `inputs`
        
        Inputs:
            inputs: tf.Tensor, 2-D tensor with size = [batch_size, feature_size]
        Returns:
            outputs: tf.Tensor, 2-D tensor with size = [batch_size, output_size]
        """
        input_shape = tuple(inputs.get_shape().as_list())
        
        if len(input_shape) != 2:
            raise base.IncompatibleShapeError("{}: rank of shape must be 2 not: {}".format(self.name, len(input_shape)))

        if input_shape[1] is None:
            raise base.IncompatibleShapeError("{}: input_size(feature_size) must be specified at module build time".format(self.name))

        if self._input_shape is not None and input_shape[1] != self._input_shape[1]:
            raise base.IncompatibleShapeError("{}: input shape must be [batch_size, {}] but not: [batch_size, {}]".format(self.name, self._input_shape[1], input_shape[1]))

        self._input_shape = input_shape

        if "w" not in self._initializers:
            self._initializers["w"] = initialization.he_normal(self._input_shape[1])
        if "b" not in self._initializers:
            self._initializers["b"] = initialization.he_normal(self._input_shape[1])

        weight_shape = (self._input_shape[1], self.output_size)
        dtype = inputs.dtype

        self._w = tf.get_variable("w", shape=weight_shape, dtype=dtype, initializer=self._initializers["w"])
        outputs = tf.matmul(inputs, self._w)
        
        if self._use_bias:
            bias_shape = (self.output_size,)
            self._b = tf.get_variable("b", shape=bias_shape, dtype=dtype, initializer=self._initializers["b"])
            otuputs = outputs + self._b
        
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
       
    @property
    def activation(self):
        return self._activation


class BatchReshape(base.AbstractModule):
    """Reshape Tensor but preserving the batch dimension."""
    
    def __init__(self, shape, name='batch_reshape'):
        """BatchReshape Constructor
        
        Inputs:
            shape: tuple/list of ints, this is the target shape besides the first dim, which is the batch_size.
                For example, we can use `shape=(-1)` to flatten a tensor.
        """
        super(BatchReshape, self).__init__(name=name)

        self._input_shape = None
        self._shape = shape

        if not callable(self._shape):
            self._shape = tuple(self._shape)

    def _infer_shape(self, dims):
        n = np.prod(dims)
        m = np.prod(abs(np.array(self._shape)))
        v = np.array(self._shape)
        v[v==-1] = n // m
        return tuple(v)

    def _build(self, inputs):
        """Connects the module into the graph, with input tensor `Input`
        
        Inputs:
            inputs: tf.Tensor, input tensor will connect to the graph.
        Outputs:
            outputs: tf.Tensor, output tensor created from this Module.
        """
        # Error checking
        if not all([isinstance(x, numbers.Integral) and (x > 0 or x ==-1) for x in self._shape]):
            raise ValueError("Target shape can only contains positive or -1 integral")
        if self._shape.count(-1) > 1:
            raise ValueError("Target shape can only have one -1 integral")

        self._input_shape = inputs.get_shape()[1:].as_list() # remove the batch dim.
        if self._shape.count(-1) > 0: # wild card -1 appears
            shape = (-1,) + self._infer_shape(self._shape)
        else:
            shape = (-1,) + self._shape

        if np.prod(shape[1:]) != np.prod(self._input_shape):
            raise ValueError("Output shape is incompatible with input shape")
        return tf.reshape(inputs, shape)

    @property
    def input_shape(self):
        self._ensure_is_connected()
        return self._input_shape


class BatchFlatten(BatchReshape):
    """Flattens the input Tensor, which is derived from `BatchReshape`"""
    def __init__(self, name='batch_flatten'):
        super(BatchFlatten,self).__init__(name=name, shape=(-1,))
