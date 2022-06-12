from lstmcnn import LSTMCNN
from keras.layers import LSTM, Conv1D, AveragePooling1D, BatchNormalization
from keras.utils import plot_model
n_features = 100
n_timesteps = 160
batch_size = 10

lstm0 = LSTM(name="lstm_1", units=32, return_sequences=True)
lstm1 = LSTM(name="lstm_2", units=32, return_sequences=True)
lstm2 = LSTM(name="lstm_3", units=32, return_sequences=True)
cnn1 = Conv1D(32, 1, activation='relu')
cnn2 = BatchNormalization()
cnn3 = AveragePooling1D(pool_size=2)


lstm_layers = [lstm0, lstm1, lstm2]
cnn_layers = [cnn1, cnn2, cnn3]
LSTM_layers = 3
shape_input = (batch_size, n_timesteps, n_features)

model = LSTMCNN(lstm_layers=lstm_layers,
                cnn_layers=cnn_layers,
                input_shape=shape_input,
                leaky_relu_alpha=0.05,
                dense_dim=1024,
                output_dim=512)
test = (n_timesteps, n_features)
model.summary()
plot_model(model.build_graph(inputs=shape_input), show_shapes=True, expand_nested=True)