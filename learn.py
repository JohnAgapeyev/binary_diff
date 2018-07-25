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
        "classes": tf.argmax(input=logits, axis=1),
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


#tf.enable_eager_execution()

#ten = []

#for arg in sys.argv:
    #data = np.fromfile(arg, np.uint8)
    #file_width = math.ceil(math.sqrt(len(data)))
    #data.resize((file_width, file_width))
    #t = tf.convert_to_tensor(data)
    #t = tf.expand_dims(t, -1)
    #t = tf.image.resize_images(t, (1024,1024), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #ten.append(t)

#dataset = tf.data.Dataset.from_tensors(ten)
#for e in dataset.make_one_shot_iterator():
    #print(e)

def get_all_files(directory):
    f = []
    for (dirpath, _, filenames) in os.walk(directory):
        for name in filenames:
            f.append(os.path.join(dirpath, name))
    return f

def data_input_fn():
    ten = []
    lab = []
    for arg in get_all_files(sys.argv[1]):
        if "Schneider" in arg:
            lab.append("Schneider")
        elif "Siemens" in arg:
            print()
            lab.append("Siemens")
        else:
            lab.append("None")
        data = np.fromfile(arg, np.uint8)
        file_width = math.ceil(math.sqrt(len(data)))
        data.resize((file_width, file_width))
        t = tf.convert_to_tensor(data)
        t = tf.expand_dims(t, -1)
        t = tf.image.resize_images(t, (1024,1024), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        ten.append(t)
    dataset = tf.data.Dataset.from_tensors((ten, lab))

    #dataset = dataset.map(parser)
    dataset = dataset.map(parser)

    dataset = dataset.shuffle(10000).batch(3)
    it = dataset.make_one_shot_iterator()
    features, labels = it.get_next()
    return features, labels

def parser(record, label):
    keys_to_features = {
        "x": tf.VarLenFeature(tf.uint8),
    }
    #parsed = tf.parse_single_example(record, keys_to_features)

    print(record)
    print(label)

    record = tf.cast(record, tf.float16)

    # Perform additional preprocessing on the parsed data.
    #image = tf.image.decode_jpeg(parsed["image_data"])
    #label = tf.cast(parsed["label"], tf.int32)
    return {"x": record}, label

classify = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/cnn_model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

#train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #x={"x": train_data},
    #y=train_labels,
    #batch_size=100,
    #num_epochs=None,
    #shuffle=True)

classify.train(
    input_fn=data_input_fn,
    steps=20000,
    hooks=[logging_hook])

#eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #x={"x": eval_data},
    #y=eval_labels,
    #num_epochs=1,
    #shuffle=False)

#eval_results = classify.evaluate(input_fn=eval_input_fn)

#print(eval_results)


