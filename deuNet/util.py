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
