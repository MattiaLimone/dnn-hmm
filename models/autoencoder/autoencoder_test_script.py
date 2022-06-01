from recurrent_autoencoder import RecurrentAutoEncoder

model = RecurrentAutoEncoder(n_encoding_layers=3, sequential_bottleneck=False, input_neurons=256, timesteps=150, n_features=39, unit_type='LSTM')
model.summary()