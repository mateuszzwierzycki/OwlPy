from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from owl_py import tbin_numpy as tbin

model_path = "C:/Users/Mateusz/PycharmProjects/OwlPy/Examples/Random100/Model/model"
save_path_weights = "C:/Users/Mateusz/PycharmProjects/OwlPy/Examples/Random100/Model/extraction_weights.tbin"
save_path_biases = "C:/Users/Mateusz/PycharmProjects/OwlPy/Examples/Random100/Model/extraction_biases.tbin"


def extract_network():
    with tf.Session() as sess:
        # restore the saved graph
        new_saver = tf.train.import_meta_graph(model_path + ".meta")
        new_saver.restore(sess, model_path)
        print("Restored the model with " + str(len(sess.graph.get_operations())) + " ops from " + model_path)

        h1_w = sess.graph.get_operation_by_name(name="h1_w").outputs[0]
        h2_w = sess.graph.get_operation_by_name(name="h2_w").outputs[0]
        lout_w = sess.graph.get_operation_by_name(name="lout_w").outputs[0]

        h1_b = sess.graph.get_operation_by_name(name="h1_b").outputs[0]
        h2_b = sess.graph.get_operation_by_name(name="h2_b").outputs[0]
        lout_b = sess.graph.get_operation_by_name(name="lout_b").outputs[0]

        weights = [h1_w.eval(None, sess), h2_w.eval(None, sess), lout_w.eval(None, sess)]
        biases = [h1_b.eval(None, sess), h2_b.eval(None, sess), lout_b.eval(None, sess)]

        print("Extracted " + str(len(weights)) + " weight tensors")
        print("Extracted " + str(len(biases)) + " bias tensors")

        tbin.save_multiple_tbin(save_path_weights, weights)
        tbin.save_multiple_tbin(save_path_biases, biases)

        print("Done")


extract_network()
