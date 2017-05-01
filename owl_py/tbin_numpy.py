"""
Supports all the IDX data types specified in the MNIST website description

TBIN file format:
int32:                total byte length of the upcoming tensor
int32:                shape count (the number of int32 values defining the shape)
int32 * shape count:  shape values
float64 * shape size: values in the array
this sequence repeats over and over for multiple arrays/tensors
you can quickly count the number of them by fast seeking the file using the first int32 of each tensor


Usage:
load_file = "C:/Users/Mateusz/Desktop/fakeout.tbin"
save_file = "C:/Users/Mateusz/Desktop/fakeout2.tbin"

loaded = load_tbin(load_file)
save_multiple_tbin(save_file, loaded, False)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy


def _write32(bytestream, value32):
    arr = numpy.fromiter([value32], dtype=numpy.int32)
    bytestream.write(arr.tobytes())


def _read32(bytestream):
    dt = numpy.dtype(numpy.int32)
    b_read = bytestream.read(4)

    if b_read == b'':
        return None, False
    else:
        return numpy.frombuffer(b_read, dtype=dt)[0], True


def _read32_in_buffer(numpy_buff, position):
    dt = numpy.dtype(numpy.int32)
    returned_position = position + 4
    return numpy.frombuffer(numpy_buff[position:position + 4], dtype=dt)[0], returned_position


def _write32_in_buffer(numpy_buff, value32, position):
    dt = numpy.dtype(numpy.int32)
    arr = numpy.fromiter([value32], dt)
    for b in arr.tobytes():
        numpy_buff[position] = b
        position += 1
    return position


def to_tbin(numpy_array):
    """Convert numpy array to a TBIN buffer representation
    Returns: A numpy array with bytes
        """
    dt = numpy.dtype(numpy.uint8)
    cnt = 4 + len(numpy_array.shape) * 4 + numpy_array.size * 8
    buff = numpy.zeros([cnt + 4], dt)
    position = int(0)
    position = _write32_in_buffer(buff, cnt, position)
    position = _write32_in_buffer(buff, len(numpy_array.shape), position)

    for val in numpy_array.shape:
        position = _write32_in_buffer(buff, val, position)

    byte_array = None

    if numpy_array.dtype == numpy.float64:
        byte_array = numpy_array.tobytes()
    else:
        casted_array = numpy_array.astype(numpy.float64)
        byte_array = casted_array.tobytes()

    byte_array = numpy.fromstring(byte_array, dt)
    buff[position:] = byte_array

    return buff


def from_tbin(byte_numpy, read_bytelength=False):
    """
    Args:
        byte_numpy: a numpy array of dtype = uint8
        read_bytelength: if byte_nympy includes the total bytelength then read it first
    Returns:
        a numpy array of dtype = float64
    """

    position = 0
    tot_len = 0
    if read_bytelength: tot_len, position = _read32_in_buffer(byte_numpy, position)
    shape_count, position = _read32_in_buffer(byte_numpy, position)
    shape_values = []
    volume = 1
    for val in range(shape_count):
        this_value, position = _read32_in_buffer(byte_numpy, position)
        shape_values.append(this_value)
        volume *= this_value

    tens = numpy.frombuffer(byte_numpy[position:position + volume * 8], dtype=numpy.float64)
    tens = tens.reshape(shape_values)
    return tens


def load_tbin(f):
    """
    Args:
        f: string file path 
    Returns:
        A list of numpy arrays of type float 64
    """

    arrays = []

    with open(f, 'rb') as bytestream:
        cnt = 0
        while True:
            tot_len, worked = _read32(bytestream)
            if not worked:
                return arrays

            arrays.append(from_tbin(bytestream.read(tot_len), False))


def save_multiple_tbin(f, numpy_arrays, stack):
    if stack:
        save_tbin(f, numpy.vstack(numpy_arrays))
    else:
        with open(f, 'wb') as bytestream:
            for tens in numpy_arrays:
                bytestream.write(to_tbin(tens).tobytes())


def save_tbin(f, numpy_array):
    """
    Args:
        f: String filepath
        numpy_array: list of numpy arrays to save 
    """
    with open(f, 'wb') as bytestream:
        bytestream.write(to_tbin(numpy_array).tobytes())
