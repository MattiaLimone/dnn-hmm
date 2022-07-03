from keras.layers import Conv1D
from keras.layers import MaxPooling1D, UpSampling1D
from models.autoencoder.convolutional_autoencoder import Convolutional1DAutoEncoder

from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as pyplot
import tensorflow as tf
import numpy as np
import pandas as pd
'''
enc_conv1 = Conv1D(12, 3, activation='relu', padding='same') <-
enc_pool1 = MaxPooling1D(2, padding='same')
enc_conv2 = Conv1D(8, 4, activation='relu', padding='same')
enc_pool2 = MaxPooling1D(4, padding='same') -> [2, 3, 4]
enc_flatten = Flatten(...) -> [2, 3*4]
enc_bottleneck = Dense(5) -> [2, 5], where 5 is the latent space

dec_dense = Dense(3*4) -> [2, 3*4]
dec_reshape = Reshape((3, 4)) -> [2, 3, 4]
dec_upsample2 = UpSampling1D(4)
dec_conv2 = Conv1DTranspose(8, 4, activation='relu', padding='same')
dec_upsample3 = UpSampling1D(2)
dec_conv3 = Conv1DTranspose(12, 3, activation='relu')
dec_output = Conv1DTranspose(1, 3, activation='sigmoid', padding='same')

encoder_layers = [enc_conv1, enc_pool1, enc_conv2, enc_ouput]
decoder_layers = [dec_conv2, dec_upsample2, dec_conv3, dec_upsample3, dec_output]
'''


def pandas_object_to_numpy_array(pandas_object):
    audio_tensor = np.zeros(
        (len(pandas_object), pandas_object.iloc[0].shape[0], pandas_object.iloc[0].shape[1]))

    for i in range(0, len(pandas_object)):
        audio_tensor[i, :, :] = pandas_object.iloc[i]

    return audio_tensor


def main():
    train = pd.read_pickle('data/cleaned/train/mel_spectr_filled_circular_train.pkl')

    train_audio_tensor_numpy = pandas_object_to_numpy_array(train['Audio_Tensor'])
    norm = np.linalg.norm(train_audio_tensor_numpy)
    #train_audio_tensor_numpy = train_audio_tensor_numpy/norm
    print(train_audio_tensor_numpy.shape)

    conv_filters = [64, 128, 256]
    conv_kernels_size = [7, 5, 5]
    conv_strides = [2, 2, 1]
    conv_pools = [2, 2, 2]
    batch_size = 10
    input_shape = train_audio_tensor_numpy.shape

    model = Convolutional1DAutoEncoder(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels_size=conv_kernels_size,
        conv_strides=conv_strides,
        latent_space_dim=1024,
        conv_pools=conv_pools,
        dropout_dense=0
    )

    # model.build()
    model.summary(expand_nested=True)

    epochs = 750
    optimizer = Adam(
        learning_rate=0.05,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam"
    )
    optimizer = SGD(
        learning_rate=0.05,
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0,
        name="SGD"
    )

    class EarlyStoppingByLossVal(Callback):
        def __init__(self, monitor='val_loss', value=0.1, verbose=0):
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
        EarlyStoppingByLossVal(monitor='val_loss', value=0.1, verbose=1),
    ]

    model.compile(optimizer=optimizer, loss='mae')

    history = model.fit(x=train_audio_tensor_numpy, epochs=epochs, batch_size=batch_size, shuffle=True,
                validation_data=(train_audio_tensor_numpy, train_audio_tensor_numpy))

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # save the encoder to file
    model.save(f'data/fitted_autoencoder/autoencoder_cnn_{epochs}_epochs')


if __name__ == "__main__":
    main()
