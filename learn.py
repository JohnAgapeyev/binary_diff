#!/bin/python3
import sys
import os
import getopt
import math
import itertools
import collections
import numpy as np
import tensorflow as tf

from PIL import Image
from multiprocessing.dummy import Pool

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 1024, 1024, 1])

    #Turns 1024x1024 to 1024x1024
    #Would be 1020x1020 if padding wasn't used
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )

    #Turns 1024x1024 to 512x512
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    #Would turns 512x512 into 508x508, but doesn't due to padding
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )

    #Turns 512x512 to 256x256
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    #Turns 256x256 to 256x256
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )

    #Turns 256x256 to 128x128
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 128*128*128])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(inputs=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


tf.enable_eager_execution()

ten = []

for arg in sys.argv:
    data = np.fromfile(arg, np.uint8)
    file_width = math.ceil(math.sqrt(len(data)))
    data.resize((file_width, file_width))
    t = tf.convert_to_tensor(data)
    t = tf.expand_dims(t, -1)
    t = tf.image.resize_images(t, (1024,1024), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ten.append(t)

dataset = tf.data.Dataset.from_tensors(ten)
for e in dataset.make_one_shot_iterator():
    print(e)

