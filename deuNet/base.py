"""Base class for DeuNet

This file contains the Abstract Base Class for defining Modules in Tensorflow.
A Module is an object which can be connected into the Graphs multiple times 
using the __call__ function, sharing variables automatically with no need to
use scope or specify resue=True.
"""
from __future__ import absoluate_import
from __future__ import division
from __future__ import print_function

import abc
import types
import six
import tensorflow as tf

#-------------------------------
# Customize some error handlers
#-------------------------------
class Error(Exception):
    """Base class for all errors from deuNet"""
    
class NotConnectedError(Error):
    """Error raise when operating on a module that has not yet been connected to graph"""

class ParentNotBuiltError(Error):
    """Error raise when the parent of a module has not been built yet"""

class IncompatibleShapeError(Error):
    """Error raise when the shape of the input at build time is not compatible"""

class NotSupportedError(Error):
    """Error raise when some operation has not been implemented yet"""



#-------------------------------
# Abstract Classes
#-------------------------------
@six.add_metaclass(abc.ABCMeta)
class AbstractModule(object):
    """Abstract Class for deuNet Modules.
    
    This class defines the functionality that every module should implement.
    We use the `tf.make_template` function to achieve weight sharing. Make sure
    use `tf.get_variable` instead of `tf.Variable` in the derived class, otherwise
    an ValueError will be thrown.
    reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.make_template.md
    """

    UPDATE_OPS_COLLECTION = tf.GraphKeys.UPDATE_OPS

    def __init__(self, name):
        if not isinstance(name, types.StringTypes):
            raise ValueError("Name must be a string.")
        self._is_connected = False
        self._template = tf.make_template(name, self._build, create_scope_now_=True)

    @abc.abstractmethod
    def _build(self,*args, **kwargs):
        """Add elements to the graph, computing output Tensors from input Tensors.

        Derived classes must implement this method, which will be wrapped into a `tf.Template` to
        achieve weight sharing.

        Inputs:
            *args: Input Tensors
            **kwargs: Additional Python Flags controlling connection
        """

    def __call__(self, *args, **kwargs):
        """This function is called automatically in module building.
        
        build the graph connection by calling `_build` function wrapped by `tf.Template`
        
        Inputs:
            *args: Input Tensors
            **kwargs: Additional Python Flags controlling connection
        Returns:
            out: `tf.tensor`, the output tensor computed by module
        """
        out = self._template(args, **kwargs)
        self._is_connected = True
        return out
    
    @property
    def name(self):
        return self._template.variable_scope.name

    @property
    def var_scope(self):
        """Returns the variable_scope declared by the module
        
        This is used as module scope for `tf.Variable` retrieval.
        Returns:
            var_scope: `tf.VariableScope` instance of this module's `tf.Template`
        """
        if not self._is_connected:
            raise NotConnectedError("Variables in {} not connected yet, __call__ the module first".format(self.name))
        return self._template.var_scope

    @property
    def is_connected(self):
        return self._is_connected

    @classmethod
    def get_possbile_initializer_keys(cls):
        """Returns the keys the dictionary of variable initializers may contain"""
        return getattr(cls, "POSSIBLE_INITIALIZER_KEYS", set())
    

class Module(AbstractModule):
    """Module is a warpper to create a deuNet module from a given function
    
    Comparing with Keras, this `Module` is like a simplified `keras.Layer`, but with 
    slightly different interface.

    example - 1-layer-dense
    ```python
    def build_model(inputs):
        x = deuNet.dense(name='linear1', activation='linear', output_size=10)(inputs)
        x = tf.nn.relu(x, name='relu1')
        y = deuNet.dense(name='linear2', activation='linear', output_size=20)(x)
        return y

    model = deuNet.Module(name='simple_dense', build=build_model)
    outputs = model(inputs)
    ```
    """

    def __init__(self, build, name="module"):
        """Construct a module with a given build function.

        Inputs:
            build: Callable to be invoked when connecting the module to the graph.
                The `build` function is invoked when the module is called, and its
                role is to specify how to add elements to the graph, and how to 
                compute output Tensors from input Tensors.
                The `build` function signature can include the following parameters:
                    *args - Input Tensors
                    **kwargs - Additional Python parameters controlling connection.
            name: Module name.
        """
        super(Module, self).__init__(name)

        if not callable(build):
            raise TypeError("Input 'build' must be callable.")
        self._build = build

    def _build(self, *args, **kwargs):
        """Forward pass the `self._build function`"""
        return self._build(*args, **kwargs)

