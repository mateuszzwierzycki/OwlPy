from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

import owl_py as owl
import owl_py.idx_numpy as idx
import owl_py.owl_data_types as types
from owl_py import communication_utils as comm

# silence the tensorflow setup
comm.set_tf_message_level(comm.MessageLevel.ERROR)

# file paths
inputs_file = "./Random100/Data/Random100_Inputs.idx"
outputs_file = "./Random100/Data/Random100_Outputs.idx"
query_file = "./Random100/Data/Random100_Query.idx"
prediction_file = "./Random100/Data/Random100_Results.idx"
model_path = "./Random100/Model/model"

# import files, have to have the same data type (float32/single int8/byte etc.)
rnd_in = owl.idx_numpy.load_idx(inputs_file)
rnd_out = owl.idx_numpy.load_idx(outputs_file)
print("Loaded the training inputs from " + inputs_file)
print("Loaded the training outputs from " + outputs_file)

# construct TensorSets for easier training (it's a very crude class right now)
tens_in = types.TensorSet(rnd_in, 80, 10, 10)  # note those tensorsets are not the same as the ones in the .net libs
tens_out = types.TensorSet(rnd_out, 80, 10, 10)  # note those tensorsets are not the same as the ones in the .net libs

# this is a file which will be used as the input array for the evaluation after training
eval_grid = idx.load_idx(query_file)
tens_eval = types.TensorSet(eval_grid, eval_grid.size / 2, 0, 0)
eval_samples = int(eval_grid.size / 2)

# get the type of data for the tensorflow code to work with
# those complex comparisons come from the numpy data type handling... not my fault.
dt = dtypes.uint8
if rnd_in.dtype.str == np.dtype(np.int8).newbyteorder(">").str: dt = dtypes.int8
if rnd_in.dtype.str == np.dtype(np.int16).newbyteorder(">").str: dt = dtypes.int16
if rnd_in.dtype.str == np.dtype(np.int32).newbyteorder(">").str: dt = dtypes.int32
if rnd_in.dtype.str == np.dtype(np.float32).newbyteorder(">").str: dt = dtypes.float32
if rnd_in.dtype.str == np.dtype(np.float64).newbyteorder(">").str: dt = dtypes.float64

# set the network size
n_inputs = int(rnd_in.size / rnd_in.shape[0])  # will resize the network automatically
n_nodes_hl1 = 20
n_nodes_hl2 = 10
n_outputs = int(rnd_out.size / rnd_out.shape[0])  # will resize the network automatically

# with this small example set we go with a very small batch
batch_size = 10

# placeholders for the data
x = tf.placeholder(dt, [None, n_inputs], name="var_x")
y = tf.placeholder(dt, [None, n_outputs], name="var_y")


# basically tensorflow network is a bunch of arrays
def network_model(data):
    h1 = {'weights': tf.Variable(tf.random_normal(shape=[n_inputs, n_nodes_hl1]), name="h1_w"),
          'biases': tf.Variable(tf.random_normal(shape=[n_nodes_hl1]), name="h1_b")}

    h2 = {'weights': tf.Variable(tf.random_normal(shape=[n_nodes_hl1, n_nodes_hl2]), name="h2_w"),
          'biases': tf.Variable(tf.random_normal(shape=[n_nodes_hl2]), name="h2_b")}

    lout = {'weights': tf.Variable(tf.random_normal(shape=[n_nodes_hl2, n_outputs]), name="lout_w"),
            'biases': tf.Variable(tf.random_normal(shape=[n_outputs]), name="lout_b")}

    l1 = tf.nn.sigmoid(tf.add(tf.matmul(data, h1['weights']), h1['biases']))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, h2['weights']), h2['biases']))
    lo = tf.nn.sigmoid(tf.add(tf.matmul(l2, lout['weights']), lout['biases']), name="prediction")

    return lo


def train_network(data):
    prediction = network_model(data)
    cost = tf.reduce_mean(tf.square(prediction - y), name="cost")
    optimizer = tf.train.RMSPropOptimizer(0.01, name="optimizer").minimize(cost)

    epochs = 300

    print("Starting the session")
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

            if epoch % 30 == 0: print("Epoch " + str(epoch) + " loss:", loss)

        saver.save(sess, model_path)
        print("Model with " + str(len(sess.graph.get_operations())) + " ops saved under " + model_path)

        # getting the network to work, it should be already trained
        eval_batch = tens_eval.next_batch(eval_samples)

        prediction_save = sess.run(prediction, feed_dict={x: eval_batch})
        prediction_save = np.asarray(prediction_save)
        prediction_save = prediction_save.reshape(eval_grid.shape[0], 1)

        # open this file and compare it with the training data outputs
        idx.save_idx(prediction_file, prediction_save)


train_network(x)
print("Done")
