"""deuNet Demo of MLP on MNIST"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

from IPython import embed

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1,ROOT)

import deuNet
import Dataset

class HParams(object):
    pass

def make_loss(model):
    with tf.variable_scope("Loss"):
        softmax_vals = tf.nn.softmax(model.outputs["score"])
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(model.outputs["score"],model.inputs["in_y"])
        loss_data = tf.reduce_mean(cross_ent)
    
    model.tracables["loss"] = loss_data
    model.tracables["softmax"] = softmax_vals
    return model

def model_builder(in_shape,label_shape):
    in_x = tf.placeholder(tf.float32, in_shape)
    in_y = tf.placeholder(tf.float32, label_shape)
    
    x = deuNet.BatchFlatten("input_flatten")(in_x)
    with tf.variable_scope('Layer1'):
        h1 = deuNet.Dense(1000, activation='relu', initializer='he_normal', name='dense1')(x)
    
    with tf.variable_scope('OutLayer'):
        score = deuNet.Dense(10, initializer='he_normal', name='outlinear')(h1)

    _inputs = {"in_x":in_x,"in_y":in_y}
    _outputs = {"score":score}
    model = deuNet.Model(_inputs, _outputs)
    model = make_loss(model)
    
    # add extra tracable tensors
    extra_tracables = {"h1":h1,"x":x}
    model.tracables.update(extra_tracables)

    return model

def train(mnist, params):
    model = model_builder(params.input_dims, params.label_dims)

    update_weights = model.trainable_weights.values()
    learning_rate = tf.placeholder(tf.float32, shape=[])

    opt = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = opt.compute_gradients(model.tracables["loss"])
    train_step = opt.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("Training Starts...\n")
    total_batch = mnist.train.num_examples // params.n_epochs
    for epoch in range(params.n_epochs):
        for batch_id in range(1,total_batch+1):
            valid_dict = {model.inputs["in_x":mnist.]}

            if batch_id % params.eval_interval == 0:


def get_params():
    params = HParams()
    params.lr = 0.5
    params.input_dims = [None, 28, 28, 1]
    params.label_dims = [None,10]
    params.batch_size = 100
    params.n_epochs = 50
    params.eval_interval = 10


if __name__ == '__main__':
    params = get_params()
    mnist = Dataset.read_data_sets()
    xs, ys = mnist.train.next_batch(batch_size = 100,one_hot=True)
    mlp_model = model_builder([None, 28,28,1],(None,10))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    feed_dict = {mlp_model.inputs['in_x']:xs, mlp_model.inputs['in_y']:ys}
    out_vals = deuNet.tf_run_sess(sess, mlp_model.outputs, feed_dict=feed_dict)
    embed()
    HERE




