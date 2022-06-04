from keras.layers import Conv1D
from keras.layers import MaxPooling1D, UpSampling1D
from models.autoencoder.convolutional_autoencoder import Convolutional1DAutoEncoder

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


def main():
    n_features = 200
    n_timesteps = 150
    batch_size = 10
    conv_filters = [12, 8]
    conv_kernels_size = [3, 4]
    conv_strides = [1, 1]
    conv_pools = [2, 2]
    input_shape = (batch_size, n_timesteps, n_features)

    model = Convolutional1DAutoEncoder(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels_size=conv_kernels_size,
        conv_strides=conv_strides,
        latent_space_dim=50,
        conv_pools=conv_pools,
        dropout_dense=0.5
    )

    # model.build()
    model.summary()


if __name__ == "__main__":
    main()
