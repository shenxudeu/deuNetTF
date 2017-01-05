"""Utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import inspect

import tensorflow as tf
from deuNet import initializations


def get_variables_in_scope(scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    """Returns a tuple of `tf.Variables` in a scope given collection.
    
    Inputs:
        scope: `tf.VariableScope` instance to get variables from.
        collection: Collection to restrict query to. By default, it will retrieve trainable variables.

    Returns:
        A tuple of `tf.Variable` objects.
    """
    scope_name = re.escape(scope.name) + '/'
    return tuple(tf.get_collection(collection, scope_name))


def get_variables_in_module(module, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    """Returns a tuple of `tf.Variable` from a `deuNet.Module`

    Inputs:
        module: `deuNet.Module` instance
        collection: Collection to restrict query to. By default, it will retrieve trainable variables.
    
    Returns:
        A tuple of `tf.Variable` objects.
    """
    return get_variables_in_scope(module.var_scope, collection=collection)


def check_initializers(initializers, keys):
    """Check the given initializers
    
    This checks that `initializers` is a dictionary that only contains keys in `keys`,
    and the entries in `initializers` are functions or dictionaries (nested).

    Inputs:
        initializers: Dictionary of initializers (allowing nested dictionaries) or None.
        keys: Iterable of valid keys for `initializers`.

    Returns:
        Copy of checked dictionary of initializers.
    """
    if initializers is None:
        return {}

    keys = set(keys)

    # check if `initializers` is a dictionary.
    if not issubclass(type(initializers), dict):
        raise TypeError("A dict of initializers was expected.")
    
    # check if extra initializers provided besides in keys
    if not set(initializers) <= keys:
        extra_keys = set(initializers) - keys
        raise KeyError("Invalid initializer keys {}, initializers can only"
                "be provided for {}".format(", ".join("'{}'".format(key) for key in extra_keys),
                                            ", ".join("'{}'".format(key) for key in keys)))
    def check_nested_callables(dictionary):
        for k,v in dictionary.iteritems():
            if isinstance(v, dict):
                check_nested_callables(v)
            elif not callable(v):
                raise TypeError("Initializer for '{}' is not a callable function or dicationary".format(k))

    check_nested_callables(initializers)
    return dict(initializers)


def _convert_activation(activation):
    """Convert activation from string to `tf.nn.callable`"""
    if activation is None:
        return None
    if isinstance(activation, str):
        return getattr(tf.nn, activation)
    else:
        raise TypeError("activation {} must be None or string".format(activation))


def check_activation(activation):
    return _convert_activation(activation)


def get_initializers(possible_keys, initializer, param_shapes, init_params=None):
    """Generate initializers for all possible parameters in the module
    
    It generates initilizers for all possible keys, then overwirte some of them from init_params. 
    Inputs:
        possible_keys: set of string , a set of parameter names need to have initilizers.
        initializer: string, the initilization methods.
        param_shapes: dictionary, shapes of all parameters.
        init_params: dict of np.array, pre-defined initilization parameters. Default is None.
    Outpus:
        initializers: dict of initializers.
    """
    # Error Checking and convert initializer to `callable`
    if initializer is None:
        initializer = "he_normal"

    if isinstance(initializer, str):
        initializer = getattr(initializations,initializer)
    else:
        raise TypeError("initializer {} must be None or string.".format(initializer))
    
    initializers = {}
    arg_len = len(getargspec(initializer)[0])
    for k in possible_keys:
        if k not in param_shapes:
            raise KeyError("parameter {} must be in param_shapes dictionary".format(k))
        if arg_len == 1: # does not need fan_in or fan_out
            initializers[k] = initializer()
        elif arg_len == 2: # need fan_in
            initializers[k] = initializer(param_shapes[k][0])
        elif arg_len == 3: # need fan_in and fan_out
            initializers[k] = initializer(param_shapes[k][0],param_shapes[k][1])
        else:
            raise ValueError("initialization func only support less then 3 args, but {} provided".format(arg_len))
           
        # overwrite parameter initializer if a initial matrix provided
        if k in init_params:
            initializers[k] = init_params[k]
    
    return initializers



