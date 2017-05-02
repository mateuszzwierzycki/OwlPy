from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from owl_py import communication_utils as comm
from owl_py import idx_numpy as idx
from owl_py import owl_data_types as types

# silence the non important messages from the tensorflow
comm.set_tf_message_level(comm.MessageLevel.ERROR)

# File paths
query_file = "./Random100/Data/Random100_Query.idx"
prediction_file = "./Random100/Data/Random100_Results_PreTrained.idx"
model_path = "./Random100/Model/model"

# this is a file which will be used as the input array for the evaluation after training
eval_grid = idx.load_idx(query_file)
tens_eval = types.TensorSet(eval_grid, eval_grid.size / 2, 0, 0)
eval_samples = int(eval_grid.size / 2)
print("Loaded the query batch from " + query_file)


def run_eval():
    with tf.Session() as sess:
        # restore the saved graph
        new_saver = tf.train.import_meta_graph(model_path + ".meta")
        new_saver.restore(sess, model_path)
        print("Restored the model with " + str(len(sess.graph.get_operations())) + " ops from " + model_path)

        # get the ops
        x = sess.graph.get_operation_by_name(name="var_x")
        prediction = sess.graph.get_operation_by_name(name="prediction")

        # get the tensors
        x = x.outputs[0]
        prediction = prediction.outputs[0]

        # predict
        print("Running the prediction")
        eval_batch = tens_eval.next_batch(eval_samples)
        prediction_save = sess.run(prediction, feed_dict={x: eval_batch})
        prediction_save = np.asarray(prediction_save)
        prediction_save = prediction_save.reshape(eval_grid.shape[0], 1)

        # save the idx file
        idx.save_idx(prediction_file, prediction_save)
        print("Saved the results as " + prediction_file)

run_eval()

print("Done")
