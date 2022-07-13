from typing import final
import keras.regularizers as regularizers
import keras.models
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adadelta
import matplotlib.pyplot as pyplot
from models.autoencoder.convolutional_autoencoder import Convolutional1DAutoEncoder
from training_utils import load_dataset, TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MEL_SPEC, coeff_determination
import tensorflow as tf


_EPOCHS_LOAD_CONV: final = 1500
_VERSION_LOAD_CONV: final = 1.5
_CONV_AUTOENC_PATH: final = f"fitted_autoencoder/cnn/autoencoder_cnn_{_EPOCHS_LOAD_CONV}_epochs_v{_VERSION_LOAD_CONV}"


def main():
    # Load dataset
    train_audio_tensor_numpy, _ = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_audio_tensor_numpy, _ = load_dataset(TEST_SET_PATH_MEL_SPEC)

    # Get input shape
    input_shape = (None, ) + train_audio_tensor_numpy.shape[1:]

    # Set model parameters
    conv_filters = [64, 128, 256]
    conv_kernels_size = [7, 5, 5]
    conv_strides = [2, 2, 1]
    conv_pools = [2, 2, 2]

    # Set model training parameters
    epochs = 2000
    batch_size = 100
    optimizer = Adadelta(
        learning_rate=1,  # was 10
        rho=0.95,
        epsilon=1e-7,
        name='adadelta_optimizer'
    )
    optimizer = SGD(
        learning_rate=0.05,
        momentum=0.9,
        nesterov=True,
        clipnorm=1,
        clipvalue=0.5,
        name="SGD"
    )

    loss = 'mae'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, min_delta=0.001, restore_best_weights=True)
    ]
    metrics = [coeff_determination]
    version = 1.5  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    retraining = int(input("Insert 0 for training and 1 for retraining: "))
    if retraining == 0:
        model = Convolutional1DAutoEncoder(
            input_shape=input_shape,
            conv_filters=conv_filters,
            conv_kernels_size=conv_kernels_size,
            conv_strides=conv_strides,
            latent_space_dim=4096,
            conv_pools=conv_pools,
            dropout_dense=0.5,
            dropout_conv=0.5
        )
    else:
        model = keras.models.load_model(_CONV_AUTOENC_PATH)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(expand_nested=True)

    # Train the model
    history = model.fit(
        x=train_audio_tensor_numpy,
        y=train_audio_tensor_numpy,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(test_audio_tensor_numpy, test_audio_tensor_numpy)
    )

    # Plot results
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Save the autoencoder to file
    if retraining == 0:
        model.save(f'fitted_autoencoder/cnn/autoencoder_cnn_{epochs}_epochs_v{version}')
    else:
        model.save(f'fitted_autoencoder/cnn/autoencoder_cnn_{epochs+_EPOCHS_LOAD_CONV}_epochs_v{version}')


if __name__ == "__main__":
    main()
