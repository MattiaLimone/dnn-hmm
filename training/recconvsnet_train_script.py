from typing import final
from keras.utils import losses_utils
from scipy.sparse import lil_matrix
import keras.models
import keras.losses
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adadelta
from matplotlib import pyplot
from models.recconvsnet.recconvsnet import RecConv1DSiameseNet
from models.autoencoder.autoencoder import ENCODER_MODEL_NAME
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    TEST_SET_PATH_MEL_SPEC, load_dataset, get_label_number, one_hot_labels_to_integer_labels


_EPOCHS_LOAD_CONV: final = 750
_EPOCHS_LOAD_REC: final = 1000
_VERSION_LOAD_CONV: final = 1.1
_VERSION_LOAD_REC: final = 1.0
_CONV_AUTOENC_PATH: final = f"fitted_autoencoder/cnn/autoencoder_cnn_{_EPOCHS_LOAD_CONV}_epochs_v{_VERSION_LOAD_CONV}"
_REC_AUTOENC_PATH: final = f"fitted_autoencoder/lstm/autoencoder_lstm_{_EPOCHS_LOAD_REC}_epochs_v{_VERSION_LOAD_REC}"


def main():
    # Load dataset and labels
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    train_mel_spec, train_mel_spec_labels = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_mel_spec, test_mel_spec_labels = load_dataset(TEST_SET_PATH_MEL_SPEC)
    total_state_number = get_label_number(train_mfccs_labels)

    # Load saved models
    conv_autoencoder = tf.keras.models.load_model(_CONV_AUTOENC_PATH)
    rec_autoencoder = tf.keras.models.load_model(_REC_AUTOENC_PATH)

    # Get input shapes
    input_shape_conv_branch = (None,) + train_mel_spec.shape[1:]
    input_shape_rec_branch = (None,) + train_mfccs.shape[1:]

    # Get recurrent and convolutional encoder layers
    conv_branch = conv_autoencoder.get_layer(ENCODER_MODEL_NAME).layers
    rec_branch = rec_autoencoder.get_layer(ENCODER_MODEL_NAME).layers

    # Set model parameters
    tail_dense_units = 2048

    # Set model training parameters
    epochs = 500
    batch_size = 250
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        name='sparse_categorical_crossentropy'
    )
    optimizer = Adadelta(
        learning_rate=1,
        rho=0.95,
        epsilon=1e-7,
        name='adadelta_optimizer'
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)
    ]
    metrics = tf.keras.metrics.SparseCategoricalAccuracy(
        name='sparse_categorical_accuracy', dtype=None
    )
    version = 0.1  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    model = RecConv1DSiameseNet(
        rec_branch_layers=rec_branch,
        conv_branch_layers=conv_branch,
        input_shape_rec_branch=input_shape_rec_branch,
        input_shape_conv_branch=input_shape_conv_branch,
        tail_dense_units=tail_dense_units,
        output_dim=total_state_number,
        tail_dense_activation='relu',
        add_repeat_vector_conv_branch=True
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(expand_nested=True)

    # Convert one-hot encoded labels to integer labels
    labels_train = one_hot_labels_to_integer_labels(train_mfccs_labels)
    labels_test = one_hot_labels_to_integer_labels(test_mfccs_labels)

    # Train the model
    history = model.fit(
        x=[train_mfccs, train_mel_spec],
        y=labels_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=([test_mfccs, test_mel_spec], labels_test)
    )

    # Plot results
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Save the model to file
    model.save(f'fitted_recconvsnet/recconvsnet_{epochs}_epochs_v{version}')


if __name__ == "__main__":
    main()
