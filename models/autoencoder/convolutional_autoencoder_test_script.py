from models.autoencoder.convolutional_autoencoder import Convolutional1DAutoEncoder


def main():
    n_features = 128
    n_timesteps = 300
    batch_size = 10
    conv_filters = [32, 64, 128, 256, 512]
    conv_kernels_size = [5, 5, 5, 3, 3]
    conv_strides = [1, 1, 1, 1, 1]
    conv_pools = [1, 1, 1, 1, 1]
    input_shape = (batch_size, n_timesteps, n_features)

    model = Convolutional1DAutoEncoder(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels_size=conv_kernels_size,
        conv_strides=conv_strides,
        latent_space_dim=64,
        conv_pools=conv_pools,
        dropout_dense=0.5
    )

    model.compile(optimizer='adam', loss='mse')
    # model.build()
    model.summary(expand_nested=True)


if __name__ == "__main__":
    main()
