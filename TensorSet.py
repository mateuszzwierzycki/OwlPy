"""A TensorSet class for organizing the data sets into parts"""
import numpy as np

class TensorSet:
    def __init__(self, numpy_array, train, test, validate):
        self._tensors = numpy_array
        self._train = train
        self._test = test
        self._validate = validate
        self._current_train = 0

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

    def next_batch(self, batch_size):
        this_batch = []

        for i in range(batch_size):
            index = int((self._current_train + i + self.train) % self.train)
            this_batch.append(self.tensors[index])

        self._current_train += batch_size
        if self._current_train > self.tensors.shape[0] : self._current_train = 0

        this_batch = np.vstack(this_batch)
        return this_batch
