from models.autoencoder.recurrent_autoencoder import RecurrentAutoEncoder
from keras import regularizers


def main():
    n_features = 128
    n_timesteps = 300
    batch_size = 10
    unit_types = ["LSTM", "LSTM"]
    recurrent_units = [1024, 512]
    activations = ["tanh", "tanh"]
    latent_space_dim = 256
    bottleneck_unit_type = "LSTM"
    bottleneck_activation = "tanh"
    recurrent_units_dropout = 0.0
    recurrent_dropout = 0.0
    input_shape = (batch_size, n_timesteps, n_features)

    model = RecurrentAutoEncoder(
        input_shape=input_shape,
        unit_types=unit_types,
        recurrent_units=recurrent_units,
        activations=activations,
        latent_space_dim=latent_space_dim,
        bottleneck_unit_type=bottleneck_unit_type,
        bottleneck_activation=bottleneck_activation,
        bottleneck_activity_regularizer=regularizers.l1(1e-4),
        recurrent_units_dropout=recurrent_units_dropout,
        recurrent_dropout=recurrent_dropout,
        bottleneck_returns_sequences=True,
        do_batch_norm=True
    )

    # model.build()
    model.summary()


if __name__ == "__main__":
    main()
