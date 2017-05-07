from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

sys.path.insert(0, "C:/Users/Mateusz/PycharmProjects/OwlPy")

import owl_py as owl
import owl_py.owl_data_types as types
from owl_py import communication_utils as comm
from owl_network import command_parser as parser

# silence the tensorflow setup
comm.set_tf_message_level(comm.MessageLevel.ERROR)

# parse the commands
commands = parser.parse_incoming("EndOfSetup")

# file paths
inputs_file = commands["inputs_file"]
outputs_file = commands["outputs_file"]
model_path = commands["model_path"]

# import files, have to have the same data type (float32/single int8/byte etc.)
parser.print_u("Loaded the training inputs from " + inputs_file)
parser.print_u("Loaded the training outputs from " + outputs_file)
rnd_in = owl.idx_numpy.load_idx(inputs_file)
rnd_out = owl.idx_numpy.load_idx(outputs_file)

# get the type of data for the tensorflow code to work with
# those complex comparisons come from the numpy data type handling... not my fault.
dt = dtypes.uint8
if rnd_in.dtype.str == np.dtype(np.int8).newbyteorder(">").str: dt = dtypes.int8
if rnd_in.dtype.str == np.dtype(np.int16).newbyteorder(">").str: dt = dtypes.int16
if rnd_in.dtype.str == np.dtype(np.int32).newbyteorder(">").str: dt = dtypes.int32
if rnd_in.dtype.str == np.dtype(np.float32).newbyteorder(">").str: dt = dtypes.float32
if rnd_in.dtype.str == np.dtype(np.float64).newbyteorder(">").str: dt = dtypes.float64

# # split into examples
# rnd_in = np.vsplit(rnd_in,rnd_in.shape[0])
# # rnd_out = np.vsplit(rnd_out,rnd_out.shape[0])
#
# for case in rnd_in:
#     case.reshape(2)
#
# for case in rnd_out:
#     case.reshape(1)

# construct TensorSets for easier training (it's a very crude class right now)
tens_in = types.TensorSet(rnd_in, 80, 10, 10)  # note those tensorsets are not the same as the ones in the .net libs
tens_out = types.TensorSet(rnd_out, 80, 10, 10)  # note those tensorsets are not the same as the ones in the .net libs

# layer counts
n_inputs = int(rnd_in[0].shape[0])  # will resize the network automatically
n_layers = int(commands["n_layers"])
n_outputs = int(rnd_out[0].shape[0])  # will resize the network automatically
n_counts = []

for i in range(n_layers):
    n_counts.append(int(commands["n_layer_" + str(i)]))

# with this small example set we go with a very small batch
batch_size = parser.try_get(commands, "batch_size", 10)

# placeholders for the data
x = tf.placeholder(dt, [None, n_inputs], name="var_x")
y = tf.placeholder(dt, [None, n_outputs], name="var_y")


# basically tensorflow network is a bunch of arrays
def network_model(data):
    layers = []
    layer_dicts = []

    h0 = {'weights': tf.Variable(tf.random_normal(shape=[n_inputs, n_counts[0]]), name="W0"),
          'biases': tf.Variable(tf.random_normal(shape=[n_counts[0]]), name="B0")}

    l0 = tf.nn.sigmoid(tf.add(tf.matmul(data, h0['weights']), h0['biases']), name="input_layer")

    layer_dicts.append(h0)
    layers.append(l0)

    prev_count = n_counts[0]

    for layer in range(n_layers):
        if layer > 0: prev_count = n_counts[layer - 1]
        this_count = n_counts[layer]

        prev_layer = layers[-1]
        this_dict = {
            'weights': tf.Variable(tf.random_normal(shape=[prev_count, this_count]), name="W" + str(layer + 1)),
            'biases': tf.Variable(tf.random_normal(shape=[this_count]), name="B" + str(layer + 1))}

        layer_dicts.append(this_dict)

        if layer < n_layers - 1:
            cur_layer = tf.nn.sigmoid(tf.add(tf.matmul(prev_layer, this_dict['weights']), this_dict['biases']),
                                      name="layer_" + str(layer))
        else:
            cur_layer = tf.nn.sigmoid(tf.add(tf.matmul(prev_layer, this_dict['weights']), this_dict['biases']),
                                      name="prediction")

        layers.append(cur_layer)

        break

    return layers[-1]


def train_network(data):
    prediction = network_model(data)
    cost = tf.reduce_mean(tf.square(prediction - y), name="cost")
    optimizer = tf.train.RMSPropOptimizer(0.01, name="optimizer").minimize(cost)

    epochs = int(parser.try_get(commands, "epochs", 1000))
    epoch_update = int(parser.try_get(commands, "epoch_update", 30))

    parser.print_u("Starting the session")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(epochs):
            loss = 0
            for batch in range(int(tens_in.train / batch_size)):
                epoch_x = tens_in.next_batch(batch_size)
                epoch_y = tens_out.next_batch(batch_size)
                batch, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                loss += c

            if epoch % epoch_update == 0:
                parser.print_u("Epoch " + str(epoch) + " loss:", loss)

        saver.save(sess, model_path)
        parser.print_u("Model with " + str(len(sess.graph.get_operations())) + " ops saved under " + model_path)


train_network(x)
