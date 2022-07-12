from typing import final
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
import keras.regularizers as regularizers
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
    tail_dense_units = 512

    # Set model training parameters
    epochs = 1

    batch_size = 1
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
        EarlyStopping(monitor='val_loss', patience=50, min_delta=0.001, restore_best_weights=True)
    ]
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(
        name='sparse_categorical_accuracy', dtype=None
    )]
    version = 0.3  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    # era regularizers.L1L2(l1=1e-04, l2=1e-03)
    # era regularizers.L1(1e-04)
    model = RecConv1DSiameseNet(
        rec_branch_layers=rec_branch,
        conv_branch_layers=conv_branch,
        input_shape_rec_branch=input_shape_rec_branch,
        input_shape_conv_branch=input_shape_conv_branch,
        tail_dense_units=tail_dense_units,
        output_dim=total_state_number,
        tail_dense_activation='relu',
        add_repeat_vector_conv_branch=True,
        kernel_regularizer_softmax=regularizers.L1(l1=0.001),
        dropout_dense=0.6
    )
    model.load_weights("fitted_recconvsnet/recconvsnet_600_epochs_v0.3")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(expand_nested=True)

    # Convert one-hot encoded labels to integer labels
    labels_train = one_hot_labels_to_integer_labels(train_mel_spec_labels)
    labels_test = one_hot_labels_to_integer_labels(test_mel_spec_labels)

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

    # Plot results loss
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Plot results accuracy
    pyplot.plot(history.history['sparse_categorical_accuracy'], label='train sparse categorical accuracy')
    pyplot.plot(history.history['val_sparse_categorical_accuracy'], label='test parse categorical accuracy')
    pyplot.legend()
    pyplot.show()

    # Save the model to file
    model.save(f'fitted_recconvsnet/recconvsnet_{epochs}_epochs_v{version}')


if __name__ == "__main__":
    main()
