"""deuNet Demo of Convolutional AutoEncoder on MNIST"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import embed

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1,ROOT)

import deuNet
import Dataset


class HParams(object):
    pass


def make_loss(model):
    loss = tf.reduce_mean(tf.square(model.outputs["decoded"] - model.inputs["in_x"]))
    model.tracables.update({"loss":loss})
    return model

def model_builder(in_shape):
    n_filters = [10, 10, 10]
    filter_sizes = [3,3,3,3]

    in_x = tf.placeholder(tf.float32, in_shape)
    x = in_x
    # Encoding
    shapes, encoder_ws = [], []
    with tf.variable_scope("Encoding"):
        for layer_i, n_output in enumerate(n_filters):
            shapes.append(x.get_shape().as_list()[1:])
            x = deuNet.Conv2D(n_filters[layer_i], (filter_sizes[layer_i],filter_sizes[layer_i]), stride=2,use_bias=True, initializer='he_normal', padding='SAME', activation='relu', name="conv{}".format(layer_i))(x)
            encoder_ws.append(deuNet.util.get_trainable_by_name("Encoding/conv{}/w".format(layer_i)))
    embedding = x
    shapes.reverse()
    encoder_ws.reverse()
    # Decoding
    for layer_i, shape in enumerate(shapes):
        n_filter = shape[-1]
        shape = shape[:-1]
        x = deuNet.Conv2DTranspose(n_filter, shape, (filter_sizes[layer_i],filter_sizes[layer_i]), stride=2,use_bias=True, initializer='he_normal',padding='SAME', activation='relu', name="deconv{}".format(layer_i), w_tensor=encoder_ws[layer_i] )(x)

    decoded = x
    
    _inputs = {"in_x":in_x}
    _outputs = {"decoded":decoded,"embedding":embedding}
    model = deuNet.Model(_inputs, _outputs)
    model = make_loss(model)
    return model


def process_epoch(sess, model, data, train_mode=False):
    """Process training/testing process for each epoch through multiple batchs"""
    num_examples = data.num_examples
    batch_size = model.params.batch_size
    total_batch = num_examples // batch_size
    if not train_mode:
        total_batch = 1 # evaluate the full batch once
    
    avg_loss = 0.
    for i in range(total_batch):
        if train_mode:
            batch_images, batch_labels = data.next_batch(batch_size,one_hot=True)
        else:
            batch_images, batch_labels = data.next_batch(batch_size,one_hot=True,full_batch=True)
        feed_dict = {model.inputs["in_x"]:batch_images, model.learning_rate:model.params.current_lr} 
        fetch_dict = model.tracables.copy()
        if train_mode:
            fetch_dict.update({"train_step":model.train_step})
        fetch_vals = deuNet.tf_run_sess(sess, fetch_dict, feed_dict)
        avg_loss += fetch_vals["loss"]
    avg_loss /= total_batch
    return avg_loss


def train(mnist, params):
    model = model_builder(params.input_dims)
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
        eval_loss = process_epoch(sess, model, mnist.valid, train_mode=False)
        train_loss = process_epoch(sess, model, mnist.train, train_mode=True)
        test_loss = process_epoch(sess, model, mnist.test, train_mode=False)
        print(deuNet.color_string("On epoch {}, validation loss = {}".format(epoch, eval_loss),'OKBLUE'))
        print(deuNet.color_string("On epoch {}, training loss = {}".format(epoch, train_loss),'OKGREEN'))
        print(deuNet.color_string("On epoch {}, testing loss = {}".format(epoch, test_loss),'FAIL'))
        print("")
        model.params.current_lr *= model.params.lr_decay

    if params.plotit:
        sample_images, _ = mnist.test.next_batch(100, one_hot=True)
        feed_dict = {model.inputs["in_x"]:sample_images[-2:], model.learning_rate:model.params.current_lr}
        fetch_dict = model.tracables.copy()
        fetch_dict.update(model.outputs)
        fetch_vals = deuNet.tf_run_sess(sess, fetch_dict, feed_dict)
        plt.imshow(fetch_vals["decoded"][0][:,:,0],cmap='gray');plt.show()

def get_params():
    params = HParams()
    params.lr = 1e-3
    params.lr_decay = .999
    params.input_dims = [None, 28, 28, 1]
    params.label_dims = [None,10]
    params.batch_size = 100
    params.n_epochs = 10
    params.eval_interval = 10
    params.keep_drop = .5
    params.plotit = True
    return params

if __name__ == "__main__":
    params = get_params()
    mnist = Dataset.read_data_sets()
    train(mnist, params)
