from typing import final
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras import regularizers
from matplotlib import pyplot

from models.autoencoder.recurrent_autoencoder import RecurrentAutoEncoder
import numpy as np
import pandas as pd


_VALIDATION_PERCENTAGE: final = 0.05
_SPARSITY_NORM_THRESHOLD: final = 1e-4


def pandas_object_to_numpy_array(pandas_object):
    audio_tensor = np.zeros(
        (len(pandas_object), pandas_object.iloc[0].shape[0], pandas_object.iloc[0].shape[1]))

    for i in range(0, len(pandas_object)):
        audio_tensor[i, :, :] = pandas_object.iloc[i]

    return audio_tensor


def main():
    train = pd.read_pickle('data/cleaned/train/mfccs_filled_circular_train.pkl')
    test = pd.read_pickle('data/cleaned/train/mfccs_filled_circular_test.pkl')
    train_audio_tensor_numpy = pandas_object_to_numpy_array(train['Audio_Tensor'])
    test_audio_tensor_numpy = pandas_object_to_numpy_array(test['Audio_Tensor'])

    batch_size = 200
    unit_types = ["LSTM" for _ in range(0, 3)]
    recurrent_units = [1024, 512, 256]
    activations = ["tanh" for _ in range(0, 3)]
    latent_space_dim = 512
    bottleneck_unit_type = "LSTM"
    bottleneck_activation = "tanh"
    recurrent_units_dropout = 0.0
    recurrent_dropout = 0.0
    input_shape = (None, ) + train_audio_tensor_numpy.shape[1:]

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

    epochs = 950

    optimizer1 = SGD(
        learning_rate=0.05,
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0,
        clipvalue=0.5,
        name="SGD"
    )

    optimizer2 = RMSprop(
        momentum=0.9,
        learning_rate=5e-2,
        clipnorm=1.0
    )

    optimizer3 = Adadelta(
        learning_rate=1,
        rho=0.95,
        epsilon=1e-07
    )

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

    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
    ]

    model.compile(optimizer=optimizer3, loss='mae')

    validation_index = int(
        train_audio_tensor_numpy.shape[0] - train_audio_tensor_numpy.shape[0] * _VALIDATION_PERCENTAGE
    )

    history = model.fit(
        x=train_audio_tensor_numpy,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(test_audio_tensor_numpy, test_audio_tensor_numpy)
    )

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Save the encoder to file
    model.save(f'data/fitted_autoencoder/autoencoder_lstm5_{epochs}_epochs')


if __name__ == "__main__":
    main()

