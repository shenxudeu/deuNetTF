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

    with tf.variable_scope("Acc"):
        correct_prediction = tf.equal(tf.argmax(model.outputs["score"],1), tf.argmax(model.inputs["in_y"],1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tracables = {"loss":loss_data, "softmax":softmax_vals, "correct_prediction":correct_prediction,"acc":acc}
    model.tracables.update(tracables)
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


def process_epoch(sess, model, data, train_mode=False):
    """Process training/testing process for each epoch through multiple batchs"""
    num_examples = data.num_examples
    batch_size = model.params.batch_size
    total_batch = num_examples // batch_size
    
    avg_loss = 0.
    avg_pred_acc = 0.
    for i in range(total_batch):
        batch_images, batch_labels = data.next_batch(batch_size,one_hot=True)
        feed_dict = {model.inputs["in_x"]:batch_images, model.inputs["in_y"]:batch_labels, model.learning_rate:model.params.current_lr} 
        fetch_dict = model.tracables
        if train_mode:
            fetch_dict.update({"train_step":model.train_step})
        fetch_vals = deuNet.tf_run_sess(sess, fetch_dict, feed_dict)
        avg_loss += fetch_vals["loss"]
        avg_pred_acc += fetch_vals["acc"]
    avg_loss /= total_batch
    avg_pred_acc /= total_batch
    return avg_loss, avg_pred_acc


def train(mnist, params):
    model = model_builder(params.input_dims, params.label_dims)
    model.params = params
    model.params.current_lr = params.lr

    update_weights = model.trainable_variables.values()
    model.learning_rate = tf.placeholder(tf.float32, shape=[])

    opt = tf.train.AdamOptimizer(model.learning_rate)
    grads_and_vars = opt.compute_gradients(model.tracables["loss"])
    model.train_step = opt.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("Training Starts...\n")
    for epoch in range(params.n_epochs):
        eval_loss, eval_acc = process_epoch(sess, model, mnist.valid, train_mode=False)
        train_loss, train_acc = process_epoch(sess, model, mnist.train, train_mode=True)
        print(deuNet.color_string("On epoch {}, validation loss = {}, validation acc. = {}".format(epoch, eval_loss, eval_acc),'OKBLUE'))
        print(deuNet.color_string("On epoch {}, training loss = {}, training acc. = {}".format(epoch, train_loss, train_acc),'OKGREEN'))
        model.params.current_lr *= model.params.lr_decay
    
    test_loss, test_acc = process_epoch(sess, model, mnist.valid, train_mode=False)
    print("On epoch {}, test loss = {}, test acc. = {}".format(epoch, test_loss, test_acc))
    
        
def get_params():
    params = HParams()
    params.lr = 1e-3
    params.lr_decay = .99
    params.input_dims = [None, 28, 28, 1]
    params.label_dims = [None,10]
    params.batch_size = 100
    params.n_epochs = 10
    params.eval_interval = 10
    return params

if __name__ == '__main__':
    params = get_params()
    mnist = Dataset.read_data_sets()
    train(mnist, params)
    #xs, ys = mnist.train.next_batch(batch_size = 100,one_hot=True)
    #mlp_model = model_builder([None, 28,28,1],(None,10))

    #init = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init)

    #feed_dict = {mlp_model.inputs['in_x']:xs, mlp_model.inputs['in_y']:ys}
    #out_vals = deuNet.tf_run_sess(sess, mlp_model.outputs, feed_dict=feed_dict)
    embed()
    HERE




