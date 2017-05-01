"""A TensorSet class for organizing the data sets into parts"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np


class TensorSet:
    def __init__(self, numpy_array, train, test, validate):
        self._tensors = numpy_array
        self._train = train
        self._test = test
        self._validate = validate
        self._current_train = 0
        self._seed = 123

    @property
    def tensors(self):
        return self._tensors

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def validate(self):
        return self._validate

    @property
    def seed(self):
        return self._seed

    # this is very very crude, needs some more work to be actually useful

    def next_batch_shuffled(self, batch_size, shuffle_seed=-1):
        this_batch = []

        for i in range(batch_size):
            index = int((self._current_train + i + self.train) % self.train)
            this_batch.append(self.tensors[index])

        self._seed = random.randint(0, 10000)
        if shuffle_seed == -1: shuffle_seed = self._seed

        random.seed(shuffle_seed)
        random.shuffle(this_batch)

        self._current_train += batch_size
        if self._current_train > self.tensors.shape[0]: self._current_train = 0

        shp = [batch_size, -1]

        this_batch = np.vstack(this_batch)
        this_batch = this_batch.reshape(shp)
        return this_batch

    def next_batch(self, batch_size):
        this_batch = []

        for i in range(batch_size):
            index = int((self._current_train + i + self.train) % self.train)
            this_batch.append(self.tensors[index])

        self._current_train += batch_size
        if self._current_train > self.tensors.shape[0] : self._current_train = 0

        shp = [batch_size, -1]

        this_batch = np.vstack(this_batch)
        this_batch = this_batch.reshape(shp)
        return this_batch
