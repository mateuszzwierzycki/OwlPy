import tensorflow as tf
import idx_numpy as idx
import numpy as np


# import files, have to have the same data type (float32/single int8/byte etc.)
rnd_in = idx.load_idx("Examples/Random100_Inputs.idx")
rnd_out= idx.load_idx("Examples/Random100_Outputs.idx")

# get the type of data for the tensorflow to work with
dt = rnd_in.dtype

# set the network size
n_inputs = rnd_in.size / rnd_in.shape[0] # will resize the network automatically
n_nodes_hl1 = 50
n_nodes_hl2 = 10
n_outputs = rnd_out.size / rnd_out[0] # will resize the network automatically

# with this small example set we go with a very small batch
batch_size = 10

def setup_graph():
    x = tf.placeholder(dt, [None,n_inputs])

