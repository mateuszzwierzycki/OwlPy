from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import owl_py as owl
from owl_py import communication_utils as comm
from owl_py import owl_data_types as types

# silence the non important messages from the tensorflow
comm.set_tf_message_level(comm.MessageLevel.ERROR)

inputs_file = "./Random100/Data/Random100_Inputs.idx"
outputs_file = "./Random100/Data/Random100_Outputs.idx"
model_path = "./Random100/Model/model"

rnd_in = owl.idx_numpy.load_idx(inputs_file)
rnd_out = owl.idx_numpy.load_idx(outputs_file)
print("Loaded the training inputs from " + inputs_file)
print("Loaded the training outputs from " + outputs_file)

# construct TensorSets for easier training (it's a very crude class right now)
tens_in = types.TensorSet(rnd_in, 80, 10, 10)  # note those tensorsets are not the same as the ones in the .net libs
tens_out = types.TensorSet(rnd_out, 80, 10, 10)  # note those tensorsets are not the same as the ones in the .net libs

# with this small example set we go with a very small batch
batch_size = 10
epochs = 1000
save_every = 100  # the model will be saved every save_every epochs
print_every = 20  # print the current loss every print_every epochs


def run_training():
    with tf.Session() as sess:

        # restore the saved graph
        new_saver = tf.train.import_meta_graph(model_path + ".meta")
        new_saver.restore(sess, model_path)
        print("Restored the model with " + str(len(sess.graph.get_operations())) + " ops from " + model_path)

        # get the ops
        x = sess.graph.get_operation_by_name(name="var_x")
        y = sess.graph.get_operation_by_name(name="var_y")
        optimizer = sess.graph.get_operation_by_name("optimizer")
        cost = sess.graph.get_operation_by_name("cost")

        # get the tensors
        x = x.outputs[0]
        y = y.outputs[0]
        cost = cost.outputs[0]

        # train
        for epoch in range(epochs):
            loss = 0
            for batch in range(int(tens_in.train / batch_size)):
                epoch_x = tens_in.next_batch(batch_size)
                epoch_y = tens_out.next_batch(batch_size)
                batch, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                loss += c

            if epoch % print_every == 0: print("Epoch " + str(epoch) + " loss:", loss)

            # save the model
            if epoch % save_every == 0 and epoch > 0:
                new_saver.save(sess, model_path)
                print("Model with " + str(len(sess.graph.get_operations())) + " ops saved under " + model_path)


run_training()
