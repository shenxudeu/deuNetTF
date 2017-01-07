"""Model Wrapper for deuceNet.

This class wraps the whole graph together, and store all trainable/non-trainable variables, input/output tensors.
When, building a real model, you can add extra tensors into the model, such as loss, scores, and etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from deuNet import base
from deuNet import util
from deuNet import initializations

from IPython import embed

class Model(object):
    """Model Class, graph wrapper"""
    
    def __init__(self, inputs, outputs):
        """Model Constructor
        
        Inputs:
            inputs: dict of tensors, <name, tensor>, this normally the input tensors/`tf.placehoder` of the graph
            outputs: dict of tensors, <name, tensor>, this could be the output tensors of graph
        """
        if type(inputs) is not dict:
            raise TypeError("model inputs must be dict")
        if type(outputs) is not dict:
            raise TypeError("model outputs must be dict")

        self.inputs = inputs
        self.outputs = outputs

        self._fetch_trainable_vars()

        self.tracables = {}

    def _fetch_trainable_vars(self):
        self._trainable_vars,self._nontrainable_vars = {}, {}
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_vars:
            self._trainable_vars[var.name] = var
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in all_vars:
            if var.name not in trainable_vars:
                self._nontrainable_vars[var.name] = var

    
    @property
    def trainable_variables(self):
        return self._trainable_vars
    
    @property
    def nontrainable_variables(self):
        return self._nontrainable_vars

        

