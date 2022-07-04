from typing import final
from keras.optimizer_v2.adadelta import Adadelta
from matplotlib import pyplot
from models.autoencoder.recurrent_autoencoder import RecurrentAutoEncoder
from training_utils import load_dataset, TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, EarlyStoppingByLossVal


_VALIDATION_PERCENTAGE: final = 0.05
_SPARSITY_NORM_THRESHOLD: final = 1e-4


def main():
    # Load dataset
    train_audio_tensor_numpy, _ = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_audio_tensor_numpy, _ = load_dataset(TEST_SET_PATH_MFCCS)

    # Get input shape
    input_shape = (None, ) + train_audio_tensor_numpy.shape[1:]

    # Set model parameters
    unit_types = ["LSTM" for _ in range(0, 3)]
    recurrent_units = [1024, 512, 256]
    activations = ["tanh" for _ in range(0, 3)]
    latent_space_dim = 512
    bottleneck_unit_type = "LSTM"
    bottleneck_activation = "tanh"
    recurrent_units_dropout = 0.0
    recurrent_dropout = 0.0

    # Set model training parameters
    epochs = 950
    batch_size = 200
    optimizer = Adadelta(
        learning_rate=1,
        rho=0.95,
        epsilon=1e-07
    )
    loss = 'mae'
    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
    ]
    version = 1.0  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    model = RecurrentAutoEncoder(
        input_shape=input_shape,
        unit_types=unit_types,
        recurrent_units=recurrent_units,
        activations=activations,
        latent_space_dim=latent_space_dim,
        bottleneck_unit_type=bottleneck_unit_type,
        bottleneck_activation=bottleneck_activation,
        recurrent_units_dropout=recurrent_units_dropout,
        recurrent_dropout=recurrent_dropout,
        bottleneck_returns_sequences=True,
        do_batch_norm=True
    )
    model.summary(expand_nested=True)
    model.compile(optimizer=optimizer, loss=loss)

    # Train the model
    history = model.fit(
        x=train_audio_tensor_numpy,
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
    model.save(f'data/fitted_autoencoder/lstm/autoencoder_lstm_{epochs}_epochs_v{version}')


if __name__ == "__main__":
    main()

