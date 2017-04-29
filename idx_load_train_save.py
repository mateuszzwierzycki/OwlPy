import tensorflow as tf
import idx_numpy as idx
import numpy as np



# samples = idxdata.read_data_set("C:/tmp/tf/datasets/inputs.idx", "C:/tmp/tf/datasets/outputs.idx", True, 100, 100, dtypes.float32, 123)
# gridsamples1 = idxdata.read_single_set("C:/tmp/tf/datasets/grid1.idx",True,dtypes.float32)
# gridsamples2 = idxdata.read_single_set("C:/tmp/tf/datasets/grid2.idx",True,dtypes.float32)
#
# n_inputs = 25 * 25
# n_nodes_hl1 = 800
# n_nodes_hl2 = 700
# n_outputs = 25 * 25
#
# batch_size = 100
#
#
# def setup_and_train(dtype=dtypes.float32):
#     setup_graph(dtype)
#
#
# def setup_graph(dtype=dtypes.float32):
#     x = tf.placeholder(dtype, [None, n_inputs])
#     y = tf.placeholder(dtype)
#
#
# def neural_network_model(data):
#     hidden_1_layer = {'weights': tf.Variable(tf.random_normal(shape=[n_inputs, n_nodes_hl1], stddev=0.03)),
#                       'biases': tf.Variable(tf.random_normal(shape=[n_nodes_hl1], stddev=0.03))}
#
#     hidden_2_layer = {'weights': tf.Variable(tf.random_normal(shape=[n_nodes_hl1, n_nodes_hl2], stddev=0.03)),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl2], stddev=0.03))}
#
#     # hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.03)),
#     #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl3],stddev=0.03))}
#
#     output_layer = {'weights': tf.Variable(tf.random_normal(shape=[n_nodes_hl2, n_outputs], stddev=0.03)),
#                     'biases': tf.Variable(tf.random_normal(shape=[n_outputs], stddev=0.03))}
#
#     l1 = tf.nn.sigmoid(tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']))
#     l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']))
#     # l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']))
#
#     output = tf.nn.sigmoid(tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases']))
#
#     return output
#
#
# def train_neural_network(x):
#     prediction = neural_network_model(x)
#     # cost = tf.reduce_mean(tf.abs(prediction -  y))
#     cost = tf.reduce_mean(tf.square(y - prediction))
#
#     # optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)
#     # optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)
#
#     optimizer = tf.train.AdamOptimizer().minimize(cost)
#
#     hm_epochs = 10000
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for _ in range(int(samples.train.num_examples / batch_size)):
#                 epoch_x, epoch_y = samples.train.next_batch(batch_size)
#                 _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                 epoch_loss += c
#             print('Epoch', epoch, "completed ouf of", hm_epochs, 'loss:', epoch_loss)
#
#             try:
#                 if (epoch % 10 == 0) & (epoch > 0):
#                     last_xcol = samples.test.input_values
#                     last_ycol = samples.test.output_values
#
#                     fpath = "C:/tmp/tf/wood/"
#
#                     for example in range(len(last_xcol)):
#                         last_x = last_xcol[example]
#                         last_y = last_ycol[example]
#
#                         input_save = last_x
#                         input_save = np.asarray(input_save)
#                         np.savetxt(fpath + "input_" + str(example) + ".csv", input_save, delimiter=", ")
#
#                         output_save = last_y
#                         output_save = np.asarray(output_save)
#                         np.savetxt(fpath + "output_" + str(example) + ".csv", output_save, delimiter=", ")
#
#                         last_x = last_x.reshape([1, 625])
#
#                         prediction_save = sess.run(prediction, feed_dict={x: last_x})
#                         prediction_save = np.asarray(prediction_save)
#                         np.savetxt(fpath + "prediction_" + str(example) + ".csv", prediction_save, delimiter=", ")
#
#                     print("Saved epoch " + str(epoch))
#
#             except:
#                 print("Failed to saved the epoch " + str(epoch))
#
#
