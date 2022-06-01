from recurrent_autoencoder import RecurrentAutoEncoder

model = RecurrentAutoEncoder(n_encoding_layers=3, input_neurons=64, timesteps=150, n_features=39, unit_type='GRU')
