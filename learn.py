#!/bin/python3
import sys
import os
import getopt
import math
import itertools
import collections
import numpy as np
import tensorflow as tf

from multiprocessing.dummy import Pool

#file_data = np.fromfile(sys.argv[1], np.uint8)
#
#print(len(file_data))
#
#file_width = math.ceil(math.sqrt(len(file_data)))
#
#file_data.resize((file_width, file_width))
#
#file_tensor = tf.convert_to_tensor(file_data)
#
#print(file_tensor)

data = np.fromfile(sys.argv[1], np.uint8)
file_width = math.ceil(math.sqrt(len(data)))
data.resize((file_width, file_width))
place_hold = tf.placeholder(data.dtype, data.shape)

#dataset = tf.data.Dataset.from_tensor_slices(place_hold)
dataset = tf.data.Dataset.from_tensors(place_hold)
iterator = dataset.make_initializable_iterator()
next_elem = iterator.get_next()




data2 = np.fromfile(sys.argv[2], np.uint8)
file_width2 = math.ceil(math.sqrt(len(data2)))
data2.resize((file_width2, file_width2))
place_hold2 = tf.placeholder(data2.dtype, data2.shape)

#dataset = tf.data.Dataset.from_tensor_slices(place_hold)
dataset2 = tf.data.Dataset.from_tensors(place_hold2)
iterator2 = dataset2.make_initializable_iterator()
next_elem2 = iterator2.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict = {place_hold: data})
    sess.run(iterator.initializer, feed_dict = {place_hold2: data2})

    dataset = dataset.concatenate(dataset2)

    t_data = sess.run(next_elem)
    print(t_data)



