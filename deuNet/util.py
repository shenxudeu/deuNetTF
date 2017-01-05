"""Utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


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

