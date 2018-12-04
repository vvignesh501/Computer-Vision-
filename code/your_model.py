#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: your_model.py

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
import tensorflow as tf
import hyperparameters as hp


class YourModel(ModelDesc):

    def __init__(self):
        super(YourModel, self).__init__()
        self.use_bias = True

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, hp.img_size, hp.img_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        num_labels = 15
        num_hidden_neurons = 1024
        hidden_weights = tf.truncated_normal([hp.img_size * hp.img_size, num_labels])
        out_weights = tf.truncated_normal([num_hidden_neurons, num_labels])
        #####################################################################
        # TASK 1: Change architecture (to try to improve performance)
        # TASK 1: Add dropout regularization

        # Declare convolutional layers
        #
        # TensorPack: Convolutional layer
        # 10 filters (out_channel), 9x9 (kernel_shape),
        # no padding, stride 1 (default settings)
        # with ReLU non-linear activation function.
        #logits = Conv2D('conv0', image, 32, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = MaxPooling('pool0', logits, (2, 2), stride=None, padding='valid')


        logits = Conv2D('conv_0', image, 64, (3,3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = Conv2D('conv0_1', logits, 64, (3,3), padding='valid', stride=(1, 1), nl=tf.nn.relu)

        #
        # TensorPack: Max pooling layer
        # Chain layers together using reference 'logits'
        # 7x7 max pool, stride = none (defaults to same as shape), padding = valid
        logits = MaxPooling('pool_0', logits, (2,2), stride=None, padding='valid')

        #logits = Conv2D('conv_00', logits, 64, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)

        #logits = MaxPooling('pool_00', logits, (2, 2), stride=None, padding='valid')

        #
        # TensorPack: Fully connected layer
        # number of outputs = number of categories (the 15 scenes in our case)
        logits = Conv2D('conv1', logits, 128, (3,3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = Conv2D('conv1_1', logits, 128, (3,3), padding='valid', stride=(1, 1), nl=tf.nn.relu)

        logits = MaxPooling('pool1', logits, (2, 2), strides=2, padding='valid')

        logits = Conv2D('conv2', logits, 256, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = Conv2D('conv2_1', logits, 256, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = Conv2D('conv2_2', logits, 256, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)

        logits = MaxPooling('pool2', logits, (2, 2), strides=2, padding='valid')

        #logits = Conv2D('conv3', logits, 256, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = Conv2D('conv3_1', logits, 256, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)
        #logits = Conv2D('conv3_2', logits, 256, (3, 3), padding='valid', stride=(1, 1), nl=tf.nn.relu)

        #logits = MaxPooling('pool3', logits, (2, 2), strides=2, padding='valid')

        keep_prob = tf.placeholder(tf.float32)

        logits = FullyConnected('fc0', logits, 4096, nl=tf.nn.relu)

        dropout=tf.nn.dropout(
            logits,
            keep_prob=0.25,
            noise_shape=None,
            seed=None,
            name=None
        )
        logits = FullyConnected('fc1', dropout, hp.category_num, nl=tf.identity)

        #####################################################################

        # Add a loss function based on our network output (logits) and the ground truth labels
        #cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)) + 0.01 * tf.nn.l2_loss(hidden_weights) + 0.01 * tf.nn.l2_loss(out_weights))


        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        #cost= cost + (0.01 * tf.nn.l2_loss(hidden_weights) + 0.01 * tf.nn.l2_loss(out_weights))

        wrong = prediction_incorrect(logits, label)

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        #####################################################################
        # TASK 1: If you like, you can add other kinds of regularization,
        # e.g., weight penalization, or weight decay

        #####################################################################

        # Set costs and monitor them for TensorBoard
        add_moving_summary(cost)
        add_param_summary(('.*/kernel', ['histogram']))  # monitor W
        self.cost = tf.add_n([cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', hp.learning_rate, summary=True)
        # Use gradient descent as our optimizer
        opt = tf.train.GradientDescentOptimizer(lr)
        # There are many other optimizers - https://www.tensorflow.org/api_guides/python/train#Optimizers
        # Including the momentum-based gradient descent discussed in class.

        return opt