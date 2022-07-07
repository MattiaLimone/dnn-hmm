import numpy as np
import pandas as pd
from keras.callbacks import Callback
from scipy import sparse as sp
import tensorflow as tf
from preprocessing.constants import AUDIO_DATAFRAME_KEY, STATE_PROB_KEY, TRAIN_SET_PATH_MFCCS, \
    TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MFCCS, TEST_SET_PATH_MEL_SPEC


def pandas_object_to_numpy_array(pandas_object) -> np.ndarray:
    audio_tensor = np.zeros(
        (len(pandas_object), pandas_object.iloc[0].shape[0], pandas_object.iloc[0].shape[1]))

    for i in range(0, len(pandas_object)):
        audio_tensor[i, :, :] = pandas_object.iloc[i]

    return audio_tensor


def load_dataset(path: str) -> (np.ndarray, list[sp.lil_matrix]):
    """
    Loads audio dataset from given path.

    :param path: path to load the dataset from.
    :return: the loaded dataset with the corresponding labels.
    """
    pandas_object_dataset = pd.read_pickle(path)
    numpy_tensor_dataset = pandas_object_to_numpy_array(pandas_object_dataset[AUDIO_DATAFRAME_KEY])
    labels_sparse_matrix_list = list(pandas_object_dataset[STATE_PROB_KEY])
    return numpy_tensor_dataset, labels_sparse_matrix_list


def get_label_number(labels: list[sp.lil_matrix]) -> int:
    return labels[0].shape[1]


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def convert_sparse_matrix_to_sparse_tensor(x: list[sp.lil_matrix]) -> tf.SparseTensor:
    labels_list = []

    for m in x:
        coo = m.tocoo()
        indices = np.pad(np.mat([coo.row, coo.col]).transpose(), (1, 0), mode='constant')
        indices = indices[1:, :]
        casted = tf.cast(tf.SparseTensor(indices, coo.data, dense_shape=(1, ) + coo.shape), dtype=tf.dtypes.int64)
        labels_list.append(casted)

    labels = tf.sparse.concat(axis=0, sp_inputs=labels_list)

    return labels


def one_hot_labels_to_integer_labels(x: list[sp.lil_matrix]) -> np.ndarray:

    labels = np.zeros(shape=(len(x), x[0].shape[0]), dtype=np.int64)

    for i in range(0, len(x)):
        # Get non-zero column values (we take into account just column indexes because there is only 1 non-zero value
        # for each row)
        labels[i, :] = x[i].nonzero()[1]

    return labels
