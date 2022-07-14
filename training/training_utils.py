from typing import Union
import keras.metrics
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from scipy import sparse as sp
import tensorflow as tf
import keras.backend as K
from preprocessing.constants import AUDIO_DATAFRAME_KEY, STATE_PROB_KEY, TRAIN_SET_PATH_MFCCS, \
    TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MFCCS, TEST_SET_PATH_MEL_SPEC, N_STATES_MFCCS


def pandas_object_to_numpy_array(pandas_object) -> np.ndarray:
    keras.metrics.sparse_top_k_categorical_accuracy()
    audio_tensor = np.zeros(
        (len(pandas_object), pandas_object.iloc[0].shape[0], pandas_object.iloc[0].shape[1]))

    for i in range(0, len(pandas_object)):
        audio_tensor[i, :, :] = pandas_object.iloc[i]

    return audio_tensor


def load_dataset(path: str, mode: int = 0) -> Union[pd.DataFrame, tuple[np.ndarray, list[sp.lil_matrix]]]:
    """
    Loads audio dataset from given path.

    :param path: path to load the dataset from.
    :param mode: if given 0 (as by default), numpy tensor containing audio features is unpacked from dataframe and
        separated from labels; if given 1 entire dataframe is given.
    :return: the loaded dataset with the corresponding labels.
    """

    pandas_object_dataset = pd.read_pickle(path)

    if mode == 0:
        numpy_tensor_dataset = pandas_object_to_numpy_array(pandas_object_dataset[AUDIO_DATAFRAME_KEY])
        labels_sparse_matrix_list = list(pandas_object_dataset[STATE_PROB_KEY])
        return numpy_tensor_dataset, labels_sparse_matrix_list
    elif mode == 1:
        return pandas_object_dataset


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


@tf.keras.utils.register_keras_serializable(package='training_utils')
def coeff_determination(y_true, y_pred):
    """
    Computes the Coefficient of Determination (also known as R^2 or R2).

    :param y_true: tensor of true targets.
    :param y_pred: tensor of predicted targets.
    :return: the Coefficient of Determination, obtained as (1 - SS_res/(SS_tot)), where
        SS_res = sum((y_true - y_pred)^2) and SS_tot = (y_true - mean(y_true)).
    """
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


@tf.__internal__.dispatch.add_dispatch_support
def sparse_top_k_categorical_speaker_accuracy(y_true, y_pred, k=5):
    """Computes how often integer targets are in the top `k` predictions.

    Standalone usage:
    >>> y_t = [2, 1]
    >>> y_p = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_t, y_p, k=3)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([1., 1.], dtype=float32)

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.

    Returns:
      Sparse top K categorical accuracy value.
    """
    # TODO: implement this

    y_pred_rank = tf.convert_to_tensor(y_pred).shape.ndims
    y_true_rank = tf.convert_to_tensor(y_true).shape.ndims

    # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            y_true = tf.reshape(y_true, [-1])

    return tf.cast(tf.compat.v1.math.in_top_k(y_pred, tf.cast(y_true, 'int32'), k), K.floatx())
