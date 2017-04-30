import numpy as np
import tensorflow as tf

import TensorSet as Owl
import communication_utils as comm
import idx_numpy as idx

# Functions to override the default log. Silence the console and print only interesting stuff.
comm.set_tf_message_level(comm.MessageLevel.ERROR)
print(comm.get_available_gpus())

# File paths
query_file = "C:/Users/Mateusz/PycharmProjects/OwlPy/Examples/Random100/Data/Random100_Query.idx"
prediction_file = "C:/Users/Mateusz/PycharmProjects/OwlPy/Examples/Random100/Data/Random100_Results_PreTrained.idx"
model_path = "C:/Users/Mateusz/PycharmProjects/OwlPy/Examples/Random100/Model/model"

# this is a file which will be used as the input array for the evaluation after training
eval_grid = idx.load_idx(query_file)
tens_eval = Owl.TensorSet(eval_grid, eval_grid.size / 2, 0, 0)
eval_samples = int(eval_grid.size / 2)


def load_full_graph(path):
    tf.global_variables_initializer()
    sess = tf.Session('', tf.Graph())
    with sess.graph.as_default():
        saver = tf.train.import_meta_graph(path + '.meta')
        saver.restore(sess, path)
        print("Initialized the graph with " + str(len(sess.graph.get_operations())) + " ops.")
    return sess


def run_eval():
    with load_full_graph(model_path) as sess:
        # print(comm.get_available_gpus())
        eval_batch = tens_eval.next_batch(eval_samples)

        x = sess.graph.get_tensor_by_name(name="var_x:0")
        prediction = sess.graph.get_tensor_by_name(name="prediction:0")

        # prediction = network_model(data)
        prediction_save = sess.run(prediction, feed_dict={x: eval_batch})
        prediction_save = np.asarray(prediction_save)
        prediction_save = prediction_save.reshape(eval_grid.shape[0], 1)
        idx.save_idx(prediction_file, prediction_save)


run_eval()
