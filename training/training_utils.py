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
@tf.keras.utils.register_keras_serializable(package='training_utils')
@tf.function
def sparse_top_k_categorical_speaker_accuracy_mfccs(y_true, y_pred, k=N_STATES_MFCCS):
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

    # Zipped tensor to iterate over two tensor simultaneously
    # zipped_tensor = tf.stack([y_pred, tf.cast(y_true, K.floatx())], axis=1)

    # Create empty output tensor to stack output for each audio
    top_k_accuracy_total_list = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

    for i in tf.range(tf.shape(y_true)[0]):

        y_pred_audio = y_pred[i]
        y_true_audio = y_true[i]

        '''
        # Checks on the shape
        y_pred_audio_rank = tf.convert_to_tensor(y_pred_audio).shape.ndims
        y_true_rank = tf.convert_to_tensor(y_true_audio).shape.ndims

        # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
        if (y_true_rank is not None) and (y_pred_audio_rank is not None):
            if y_pred_audio_rank > 2:
                y_pred_audio = tf.reshape(y_pred_audio, [-1, y_pred_audio.shape[-1]])
            if y_true_rank > 1:
                y_true_audio = tf.reshape(y_true_audio, [-1])
        '''

        # Check the right range, looking the first y_true_audio element
        first_state = y_true_audio[0]

        start_range = tf.subtract(
            first_state,
            tf.math.mod(first_state, tf.convert_to_tensor(N_STATES_MFCCS))
        )

        end_range = tf.add(start_range, tf.convert_to_tensor(N_STATES_MFCCS))

        top_k_accuracy_audio = tf.fill(tf.shape(y_true_audio), tf.convert_to_tensor(False))
        # For each state
        for state in tf.range(start_range, end_range):
            y_pred_audio_state = tf.fill(tf.shape(y_true_audio), state)

            # Calculate top_k_accuracy vector for given audio and apply logical OR with all the other states
            top_k_accuracy_audio = tf.math.logical_or(
                top_k_accuracy_audio,
                tf.compat.v1.math.in_top_k(y_pred_audio, tf.cast(y_pred_audio_state, 'int32'), k)
            )
        # At the end of this loop, the top_k_accuracy_audio vector will contain True in i-th position if at least one
        # valid range state in the top-k most probable ones, so stack this result over the ones of the other audios
        top_k_accuracy_total_list = top_k_accuracy_total_list.write(
            index=top_k_accuracy_total_list.size(),
            value=top_k_accuracy_audio
        )

    top_k_accuracy_total = top_k_accuracy_total_list.stack()

    return tf.cast(top_k_accuracy_total, K.floatx())


@tf.__internal__.dispatch.add_dispatch_support
@tf.keras.utils.register_keras_serializable(package='training_utils')
@tf.function
def sparse_categorical_speaker_accuracy_mfccs(y_true, y_pred, k=N_STATES_MFCCS):
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

    # Zipped tensor to iterate over two tensor simultaneously
    # zipped_tensor = tf.stack([y_pred, tf.cast(y_true, K.floatx())], axis=1)

    # Create empty output tensor to stack output for each audio
    speaker_state_accuracy_total_list = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

    for i in tf.range(tf.shape(y_true)[0]):

        y_pred_audio = y_pred[i]
        y_true_audio = y_true[i]

        '''
        # Checks on the shape
        y_pred_audio_rank = tf.convert_to_tensor(y_pred_audio).shape.ndims
        y_true_rank = tf.convert_to_tensor(y_true_audio).shape.ndims

        # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
        if (y_true_rank is not None) and (y_pred_audio_rank is not None):
            if y_pred_audio_rank > 2:
                y_pred_audio = tf.reshape(y_pred_audio, [-1, y_pred_audio.shape[-1]])
            if y_true_rank > 1:
                y_true_audio = tf.reshape(y_true_audio, [-1])
        '''

        # Check the right range, looking the first y_true_audio element
        first_state = y_true_audio[0]
        start_range = tf.subtract(
            first_state,
            tf.math.mod(first_state, tf.convert_to_tensor(N_STATES_MFCCS))
        )

        end_range = tf.add(start_range, tf.convert_to_tensor(N_STATES_MFCCS))

        speaker_state_accuracy_audio = tf.fill(tf.shape(y_true_audio), tf.convert_to_tensor(False))
        # For each state
        for state in tf.range(start_range, end_range):
            y_pred_audio_state = tf.fill(tf.shape(y_true_audio), state)

            # Calculate top_1_accuracy vector for given audio and apply logical OR with all the other states
            speaker_state_accuracy_audio = tf.math.logical_or(
                speaker_state_accuracy_audio,
                tf.compat.v1.math.in_top_k(y_pred_audio, tf.cast(y_pred_audio_state, 'int32'), 1)
            )
        # At the end of this loop, the speaker_state_accuracy_audio vector will contain True in i-th position if at
        # least one valid range state in the top-k most probable ones, so stack this result over the ones of the other
        # audios
        speaker_state_accuracy_total_list.write(
            index=speaker_state_accuracy_total_list.size(),
            value=speaker_state_accuracy_audio
        )

    top_k_accuracy_total = speaker_state_accuracy_total_list.stack()

    return tf.cast(top_k_accuracy_total, K.floatx())


@tf.__internal__.dispatch.add_dispatch_support
@tf.keras.utils.register_keras_serializable(package='training_utils')
@tf.function
def speaker_n_states_in_top_k_accuracy_mfccs(y_true, y_pred):
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

    # Zipped tensor to iterate over two tensor simultaneously
    # zipped_tensor = tf.stack([y_pred, tf.cast(y_true, K.floatx())], axis=1)

    # Create empty list to stack output for each audio
    top_k_accuracy_total_list = tf.TensorArray(dtype=K.floatx(), size=0, dynamic_size=True)

    for i in tf.range(tf.shape(y_true)[0]):

        y_pred_audio = y_pred[i]
        y_true_audio = y_true[i]

        '''
        # Checks on the shape
        y_pred_audio_rank = tf.convert_to_tensor(y_pred_audio).shape.ndims
        y_true_rank = tf.convert_to_tensor(y_true_audio).shape.ndims

        # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
        if (y_true_rank is not None) and (y_pred_audio_rank is not None):
            if y_pred_audio_rank > 2:
                y_pred_audio = tf.reshape(y_pred_audio, [-1, y_pred_audio.shape[-1]])
            if y_true_rank > 1:
                y_true_audio = tf.reshape(y_true_audio, [-1])
        '''

        # Check the right range, looking the first y_true_audio element
        first_state = y_true_audio[0]
        start_range = tf.subtract(
            first_state,
            tf.math.mod(first_state, tf.convert_to_tensor(N_STATES_MFCCS))
        )
        end_range = tf.add(start_range, tf.convert_to_tensor(N_STATES_MFCCS))

        top_k_accuracy_audio = tf.fill(tf.shape(y_true_audio), tf.convert_to_tensor(0, dtype=K.floatx()))
        # For each state
        for state in tf.range(start_range, end_range):
            y_pred_audio_state = tf.fill(tf.shape(y_true_audio), state)

            # Calculate top_k_accuracy vector for given audio and apply logical OR with all the other states
            top_k_accuracy_audio = tf.add(
                top_k_accuracy_audio,
                tf.cast(tf.compat.v1.math.in_top_k(y_pred_audio, tf.cast(y_pred_audio_state, 'int32'), N_STATES_MFCCS),
                        K.floatx())
            )

        # At the end of this loop, the top_k_accuracy_audio vector will contain the number x of valid range states
        # in the top-k most probable ones (not considering repeating probabilities, N_STATES_MFCCS in the best case, 0
        # in the worst case), then divide the result  by N_STATES_MFCCS and take the minimum between 1 and the result
        # (to get 1 in the best case, considering also repeating probabilities in the top-k can cause the result of the
        # division to go above 1) and stack it over the ones of the other audios
        top_k_accuracy_audio = tf.minimum(
            tf.divide(top_k_accuracy_audio, tf.convert_to_tensor(N_STATES_MFCCS, dtype=K.floatx())),
            tf.fill(tf.shape(y_true_audio), tf.convert_to_tensor(1.0, dtype=K.floatx()))
        )
        top_k_accuracy_total_list = top_k_accuracy_total_list.write(
            index=top_k_accuracy_total_list.size(),
            value=top_k_accuracy_audio
        )

    top_k_accuracy_total = top_k_accuracy_total_list.stack()

    return tf.cast(top_k_accuracy_total, K.floatx())


