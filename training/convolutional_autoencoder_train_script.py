from models.autoencoder.convolutional_autoencoder import Convolutional1DAutoEncoder
from keras.optimizer_v2.gradient_descent import SGD
import matplotlib.pyplot as pyplot
from training_utils import load_dataset, TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MEL_SPEC, EarlyStoppingByLossVal


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
    epochs = 750
    batch_size = 10
    optimizer = SGD(
        learning_rate=0.05,
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0,
        name="SGD"
    )
    loss = 'mae'
    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.1, verbose=1),
    ]
    version = 1.1  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    model = Convolutional1DAutoEncoder(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels_size=conv_kernels_size,
        conv_strides=conv_strides,
        latent_space_dim=1024,
        conv_pools=conv_pools,
        dropout_dense=0
    )
    model.compile(optimizer=optimizer, loss=loss)
    model.summary(expand_nested=True)

    # Train the model
    history = model.fit(
        x=train_audio_tensor_numpy,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(train_audio_tensor_numpy, test_audio_tensor_numpy)
    )

    # Plot results
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Save the autoencoder to file
    model.save(f'data/fitted_autoencoder/cnn/autoencoder_cnn_{epochs}_epochs_v{version}')


if __name__ == "__main__":
    main()
