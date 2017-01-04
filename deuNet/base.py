"""Base class for DeuNet

This file contains the Abstract Base Class for defining Modules in Tensorflow.
A Module is an object which can be connected into the Graphs multiple times 
using the __call__ function, sharing variables automatically with no need to
use scope or specify resue=True.
"""
from __future__ import absoluate_import
from __future__ import division
from __future__ import print_function

import abs
import types
import six
import tensorflow as tf


