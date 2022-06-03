from keras.layers import Conv1D
from keras.layers import MaxPooling1D, UpSampling1D

from models.autoencoder.convolutional_autoencoder import ConvolutionalAutoEncoder

enc_conv1 = Conv1D(12, 3, activation='relu', padding='same')
enc_pool1 = MaxPooling1D(2, padding='same')
enc_conv2 = Conv1D(8, 4, activation='relu', padding='same')
enc_ouput = MaxPooling1D(4, padding='same')

dec_conv2 = Conv1D(8, 4, activation='relu', padding='same')
dec_upsample2 = UpSampling1D(4)
dec_conv3 = Conv1D(12, 3, activation='relu')
dec_upsample3 = UpSampling1D(2)
dec_output = Conv1D(1, 3, activation='sigmoid', padding='same')

encoder_layers = [enc_conv1, enc_pool1, enc_conv2, enc_ouput]
decoder_layers = [dec_conv2, dec_upsample2, dec_conv3, dec_upsample3, dec_output]

model = ConvolutionalAutoEncoder(
        n_features=39,
        encoder_layers=encoder_layers,
        bottleneck=enc_ouput,
        decoder_layers=decoder_layers,
        outputs_sequences=True,
        input_shape=(32, 150, 39),
    )

# model.build()
model.summary()



