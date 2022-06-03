from keras.layers import ConvLSTM1D, LSTM
from keras.layers import MaxPooling1D, UpSampling1D

from models.autoencoder.convolutional_recurrent_autoencoder import ConvolutionalRecurrentAutoEncoder

enc_conv1 = ConvLSTM1D(12, 3, activation='relu', padding='same')
enc_pool1 = MaxPooling1D(2, padding='same')
enc_conv2 = ConvLSTM1D(8, 4, activation='relu', padding='same')
enc_ouput = MaxPooling1D(4, padding='same')

lstm2 = LSTM(name='Bottleneck', units=64, activation='relu', return_sequences=True)

dec_conv2 = ConvLSTM1D(8, 4, activation='relu', padding='same')
dec_upsample2 = UpSampling1D(4)
dec_conv3 = ConvLSTM1D(12, 3, activation='relu')
dec_upsample3 = UpSampling1D(2)
dec_output = ConvLSTM1D(1, 3, activation='sigmoid', padding='same')

encoder_layers = [enc_conv1, enc_pool1, enc_conv2, enc_ouput]
decoder_layers = [dec_conv2, dec_upsample2, dec_conv3, dec_upsample3, dec_output]

model = ConvolutionalRecurrentAutoEncoder(
        n_features=39,
        encoder_layers=encoder_layers,
        bottleneck=lstm2,
        decoder_layers=decoder_layers,
        outputs_sequences=True,
        input_shape=(32, 150, 39),
    )

# model.build()
model.summary()



