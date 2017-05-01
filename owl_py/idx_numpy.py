"""Supports all the IDX data types specified in the MNIST website description"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy


def _read32(bytestream):
    dt = numpy.dtype(numpy.int32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _write32(bytestream, value32):
    dt = numpy.dtype(numpy.int32).newbyteorder('>')
    arr = numpy.fromiter([value32], dt)
    bytestream.write(arr.tobytes())

def save_idx(f, numpy_array):
    with open(f, 'wb') as bytestream:

        # magic
        bytestream.write(bytes([0]))
        bytestream.write(bytes([0]))

        swap_bytes = False

        if numpy_array.dtype.str == numpy.dtype(numpy.uint8).newbyteorder(">").str: bytestream.write(bytes([8]))
        if numpy_array.dtype.str == numpy.dtype(numpy.int8).newbyteorder(">").str: bytestream.write(bytes([9]))
        if numpy_array.dtype.str == numpy.dtype(numpy.int16).newbyteorder(">").str: bytestream.write(bytes([11]))
        if numpy_array.dtype.str == numpy.dtype(numpy.int32).newbyteorder(">").str: bytestream.write(bytes([12]))
        if numpy_array.dtype.str == numpy.dtype(numpy.float32).newbyteorder(">").str: bytestream.write(bytes([13]))
        if numpy_array.dtype.str == numpy.dtype(numpy.float64).newbyteorder(">").str: bytestream.write(bytes([14]))

        if numpy_array.dtype.str == numpy.dtype(numpy.uint8).str:
            bytestream.write(bytes([8]))
            swap_bytes = True
        if numpy_array.dtype.str == numpy.dtype(numpy.int8).str:
            bytestream.write(bytes([9]))
            swap_bytes = True
        if numpy_array.dtype.str == numpy.dtype(numpy.int16).str:
            bytestream.write(bytes([11]))
            swap_bytes = True
        if numpy_array.dtype.str == numpy.dtype(numpy.int32).str:
            bytestream.write(bytes([12]))
            swap_bytes = True
        if numpy_array.dtype.str == numpy.dtype(numpy.float32).str:
            bytestream.write(bytes([13]))
            swap_bytes = True
        if numpy_array.dtype.str == numpy.dtype(numpy.float64).str:
            bytestream.write(bytes([14]))
            swap_bytes = True

        bytestream.write(bytes([len(numpy_array.shape)]))

        # shape
        for val in numpy_array.shape:
            _write32(bytestream, val)

        if swap_bytes:
            revd = numpy_array.byteswap(False)
            bytestream.write(revd.tobytes())
        else:
            bytestream.write(numpy_array.tobytes())


def load_idx(f):
    with open(f, 'rb') as bytestream:
        buff = bytestream.read(4)

        dt = numpy.dtype(numpy.uint8)
        if buff[2] == 0x09: dt = numpy.dtype(numpy.int8)
        if buff[2] == 0x0B: dt = numpy.dtype(numpy.int16)
        if buff[2] == 0x0C: dt = numpy.dtype(numpy.int32)
        if buff[2] == 0x0D: dt = numpy.dtype(numpy.float32)
        if buff[2] == 0x0E: dt = numpy.dtype(numpy.float64)

        dimensions = buff[3]
        tot_len = 1
        dims = []

        for val in range(0, dimensions):
            dims.append(_read32(bytestream))
            tot_len *= dims[len(dims) - 1]

        buff = bytestream.read(tot_len * dt.itemsize)

        dt = dt.newbyteorder('>')
        data = numpy.frombuffer(buff, dt)
        data = data.reshape(dims)

        return data
