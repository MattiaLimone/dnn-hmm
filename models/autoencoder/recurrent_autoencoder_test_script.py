from keras.layers import LSTM, GRU
from models.autoencoder.recurrent_autoencoder import RecurrentAutoEncoder, LSTMRepeatVector, GRURepeatVector
from keras import regularizers, Sequential


def main():
    n_features = 39
    n_timesteps = 243
    batch_size = 200
    unit_types = ["LSTM", "LSTM"]
    recurrent_units = [1024, 512]
    activations = ["tanh", "tanh"]
    recurrent_regularizers = [regularizers.l1(1e-4), None]
    kernel_regularizers = [regularizers.l1(1e-4), None]
    bias_regularizers = [regularizers.l1(1e-4), None]
    activity_regularizers = [regularizers.l1(1e-4), None]
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
        bottleneck_recurrent_regularizer=regularizers.l1(1e-4),
        bottleneck_kernel_regularizer=regularizers.l1(1e-4),
        bottleneck_bias_regularizer=regularizers.l1(1e-4),
        bottleneck_activity_regularizer=regularizers.l1(1e-4),
        recurrent_regularizer=recurrent_regularizers,
        kernel_regularizer=kernel_regularizers,
        bias_regularizer=bias_regularizers,
        activity_regularizer=activity_regularizers,
        recurrent_units_dropout=recurrent_units_dropout,
        recurrent_dropout=recurrent_dropout,
        bottleneck_returns_sequences=True,
        do_batch_norm=True,
        last_layer_bias_regularizer=regularizers.l1(1e-4),
        last_layer_activity_regularizer=regularizers.l1(1e-4),
        last_layer_kernel_regularizer=regularizers.l1(1e-4),
        last_layer_activation='relu'
    )
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # get_config() test
    config = model.get_config()
    model = RecurrentAutoEncoder.from_config(config)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # LSTMRepeatVector get_config() test
    timesteps = input_shape[1]
    rec_repeat_vector = LSTMRepeatVector(
        units=20,
        repeat_vector_timesteps=timesteps
    )
    model = Sequential()
    model.add(LSTM(units=40, activation='tanh', return_sequences=True))
    model.add(LSTM(units=30, activation='tanh', return_sequences=True))
    model.add(rec_repeat_vector)
    model.build(input_shape)
    model.summary(expand_nested=True)

    rec_repeat_vector = LSTMRepeatVector.from_config(rec_repeat_vector.get_config())
    model = Sequential()
    model.add(LSTM(units=40, activation='tanh', return_sequences=True))
    model.add(LSTM(units=30, activation='tanh', return_sequences=True))
    model.add(rec_repeat_vector)
    model.build(input_shape)
    model.summary(expand_nested=True)

    # GRURepeatVector get_config() test
    rec_repeat_vector = GRURepeatVector(
        units=20,
        repeat_vector_timesteps=timesteps
    )
    model = Sequential()
    model.add(GRU(units=40, activation='tanh', return_sequences=True, reset_after=True))
    model.add(GRU(units=30, activation='tanh', return_sequences=True, reset_after=True))
    model.add(rec_repeat_vector)
    model.build(input_shape)
    model.summary(expand_nested=True)

    rec_repeat_vector = GRURepeatVector.from_config(rec_repeat_vector.get_config())
    model = Sequential()
    model.add(GRU(units=40, activation='tanh', return_sequences=True, reset_after=True))
    model.add(GRU(units=30, activation='tanh', return_sequences=True, reset_after=True))
    model.add(rec_repeat_vector)
    model.build(input_shape)
    model.summary(expand_nested=True)


if __name__ == "__main__":
    main()
