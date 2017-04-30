import tensorflow as tf
import idx_numpy as idx
import numpy as np
import TensorSet as Owl

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
    return sess


def run_eval():
    with load_full_graph(model_path) as sess:
        eval_batch = tens_eval.next_batch(eval_samples)

        x = sess.graph.get_tensor_by_name(name="var_x:0")
        prediction = sess.graph.get_tensor_by_name(name="prediction:0")

        # prediction = network_model(data)
        prediction_save = sess.run(prediction, feed_dict={x: eval_batch})
        prediction_save = np.asarray(prediction_save)
        prediction_save = prediction_save.reshape(eval_grid.shape[0], 1)
        idx.save_idx(prediction_file, prediction_save)

run_eval()
