#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: vgg_model.py

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
import tensorflow as tf
import hyperparameters as hp

class VGGModel(ModelDesc):

    def __init__(self):
        super(VGGModel, self).__init__()
        self.activation_fn = tf.nn.relu
        self.conv_padding = 'SAME'
        self.pool_padding = 'SAME'
        self.use_bias = True

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        ################################################################################
        # TASK 2: Fine tuning
        #
        # We wish to replace VGG's last fully connected layer. 
        # We need to change the number of classes for our output, too.
        #
        # Each layer has a name. TensorPack will load the weights from vgg16.npy and 
        # match the names of the specified layers to the layers which exist in vgg16.npy.
        # In this case, we need to _use a different name than fc8_, otherwise TensorPack
        # will copy over the existing weights.
        #
        # Training will take _a long time_ - two hours per epoch on my laptop CPU. Run it
        # overnight, and use the feature to continue from where you left off (per epoch).
        # You'll notice it once you've executed run.py multiple times.
        #
        # Weight freezing: It is also possible to stop gradient propagation beyond the 
        # newly added fc layer. It is not required, but please feel free to investigate
        # this via tf.stop_gradient
        #
        ################################################################################

        # TensorPack: This is a slightly different notation for the network architecture
        # It pre-declares variables for all Conv2D layers (argscope).
        #

        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
            logits = (LinearWrap(image)
                      .Conv2D('conv1_1', 64)
                      .Conv2D('conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 112
                      .Conv2D('conv2_1', 128)
                      .Conv2D('conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 56
                      .Conv2D('conv3_1', 256)
                      .Conv2D('conv3_2', 256)
                      .Conv2D('conv3_3', 256)
                      .MaxPooling('pool3', 2)
                      # 28
                      .Conv2D('conv4_1', 512)
                      .Conv2D('conv4_2', 512)
                      .Conv2D('conv4_3', 512)
                      .MaxPooling('pool4', 2)
                      # 14
                      .Conv2D('conv5_1', 512)
                      .Conv2D('conv5_2', 512)
                      .Conv2D('conv5_3', 512)
                      .MaxPooling('pool5', 2)
                      # 7

                      .FullyConnected('fc6_new', 4096, nl=tf.nn.relu)
                      .FullyConnected('fc7_new', 4096, nl=tf.nn.relu)
                      .FullyConnected('fc8_new', out_dim=15, nl=tf.identity)()
                      )
        #logits = tf.stop_gradient(logits)
        #logits=tf.reshape(logits,[5,25088])
        #keep_prob = tf.placeholder(tf.float32)
        #logits_2=  tf.contrib.layers.fully_connected(logits, 4096, tf.nn.relu)
        #logits_2 = tf.contrib.layers.fully_connected( logits_2, 4096, tf.nn.relu)
        #logits_2 = tf.contrib.layers.fully_connected(logits_2, 15, tf.identity)

        #logits_2 = (logits.FullyConnected('fc6_new', logits_2, 4096, nl=tf.nn.relu)
         #               .FullyConnected('fc7_new', 4096, nl=tf.nn.relu)
         #               .FullyConnected('fc8_new', out_dim=15, nl=tf.identity)())

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        add_moving_summary(cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', hp.learning_rate, summary=True)
        opt = tf.train.RMSPropOptimizer(lr)
        return opt

