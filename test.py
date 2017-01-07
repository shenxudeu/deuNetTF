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

def test_dense():
    x_data = np.random.randn(100, 5)
    x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]], name='x')
    init_params = {'w':np.zeros((5,20),dtype=np.float32),'b':np.ones((20,),dtype=np.float32)}
    with tf.name_scope("layer1"):
        dense1 = deuNet.Dense(20, use_bias=True, initializer='glorot_normal', name='dense1',initial_params=init_params,activation='relu')
        x_1 = dense1(x)

    with tf.name_scope("layer2"):
        dense2 = deuNet.Dense(40, use_bias=True, initializer='glorot_normal', name='dense2',activation=None)
        y = dense2(x_1)

    www = tf.Variable(name='www',initial_value=np.zeros((10,1)))
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    feed_dict = {x:x_data}
    y_vals = sess.run(y, feed_dict=feed_dict)

    print("Linear output is ", y_vals)
    print("module 1: {}".format(dense1.var_scope.name))
    print("\t trainable variables: {}".format(['{} ,'.format(k.name) for k in deuNet.get_variables_in_module(dense1)]))

    print("module 2: {}".format(dense2.var_scope.name))
    print("\t trainable variables: {}".format(['{} ,'.format(k.name) for k in deuNet.get_variables_in_module(dense2)]))
    print("\n\n\n")
    embed()


if __name__ == '__main__':
    #weight_sharing()
    #foo()
    test_dense()
