"""Debugging file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from IPython import embed

import deuNet

def foo():
    x_data = np.random.randn(100).astype(np.float32)
    y_data = x_data * 0.1 - 0.12
    
    with tf.name_scope('weightsL1'):
        W = tf.Variable(tf.random_uniform([1], -1., 1.),name='W')
        b = tf.Variable(tf.random_uniform([1],-0.01, 0.01),name='bias')

    with tf.name_scope('weightsL2'):
        W2 = tf.Variable(tf.random_uniform([1], -1., 1.),name='W')
    y = W * x_data + b
    y = W2 * y

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.AdamOptimizer(0.05)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(200):
        sess.run(train)
        if i % 20 == 0:
            print(i, sess.run(W), sess.run(b), sess.run(loss))
    embed()

def weight_sharing():
    def scale_op(x, scalar_name):
        var1 = tf.get_variable(scalar_name, shape=[], initializer=tf.constant_initializer(1))
        return x * var1

    scale_by_y = tf.make_template('scale_by_y', scale_op, scalar_name='y')
    
    x1_data = np.random.randn(10).astype(np.float32)
    x2_data = np.random.randn(5).astype(np.float32)

    y1 = scale_by_y(x1_data)
    y2 = scale_by_y(x2_data)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("y1 = ", sess.run(y1))
    print("y2 = ", sess.run(y2))
    embed()


if __name__ == '__main__':
    weight_sharing()
    #foo()
