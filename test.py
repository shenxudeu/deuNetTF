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
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(200):
        sess.run(train)
        if i % 20 == 0:
            print(i, sess.run(W), sess.run(b), sess.run(loss))
    embed()

if __name__ == '__main__':
    foo()
