from models.autoencoder.convolutional_autoencoder import Convolutional1DAutoEncoder


def main():
    n_features = 128
    n_timesteps = 227
    batch_size = 200
    conv_filters = [32, 64, 128]
    conv_kernels_size = [5, 5, 3]
    conv_strides = [2, 2, 2]
    conv_pools = [2, 2, 2]
    input_shape = (batch_size, n_timesteps, n_features)

    model = Convolutional1DAutoEncoder(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels_size=conv_kernels_size,
        conv_strides=conv_strides,
        latent_space_dim=64,
        conv_pools=conv_pools,
        dropout_dense=0.5,
        do_batch_norm=True,
        last_layer_activation='relu'
    )
    model.compile(optimizer='adam', loss='mse')
    model.summary(expand_nested=True)

    # get_config() test
    config = model.get_config()
    model = Convolutional1DAutoEncoder.from_config(config)
    model.compile(optimizer='adam', loss='mse')
    model.summary(expand_nested=True)


if __name__ == "__main__":
    main()
