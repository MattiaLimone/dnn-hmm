from keras import Sequential
from autoencoder import AutoEncoder, FlattenDenseLayer
from keras.layers import LSTM


def main():

    input_shape = (10, 150, 39)
    lstm0 = LSTM(name="lstm_encoder1", units=512, activation='tanh', return_sequences=True)
    lstm1 = LSTM(name="lstm_encoder2", units=256, activation='tanh', return_sequences=True)
    lstm2 = LSTM(name="lstm_encoder3", units=128, activation='tanh', return_sequences=True)
    lstm3 = LSTM(name='Bottleneck', units=64, activation='tanh', return_sequences=True)
    lstm_decoder0 = LSTM(name="lstm_decoder1", units=128, activation='tanh', return_sequences=True)
    lstm_decoder1 = LSTM(name="lstm_decoder2", units=256,  activation='tanh', return_sequences=True)
    lstm_decoder2 = LSTM(name="lstm_decoder3", units=512,  activation='tanh', return_sequences=True)
    encoder_layers = [lstm0, lstm1, lstm2]
    decoder_layers = [lstm_decoder0, lstm_decoder1, lstm_decoder2]
    model = AutoEncoder(
        encoder_layers=encoder_layers,
        bottleneck=lstm3,
        decoder_layers=decoder_layers,
        outputs_sequences=True,
        do_batch_norm=True,
        input_shape=input_shape,
    )
    model.summary(expand_nested=True)

    # get_config() test
    config = model.get_config()
    model = AutoEncoder.from_config(config)
    model.summary(expand_nested=True)

    # FlattenDenseLayer get_config() test
    flatten_dense = FlattenDenseLayer(units=10, dropout=0.5)
    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(LSTM(units=25, activation='tanh', return_sequences=True))
    model.add(flatten_dense)
    model.build(input_shape)
    model.summary(expand_nested=True)

    flatten_dense = FlattenDenseLayer.from_config(flatten_dense.get_config())
    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(LSTM(units=25, activation='tanh', return_sequences=True))
    model.add(flatten_dense)
    model.build(input_shape)
    model.summary(expand_nested=True)


if __name__ == "__main__":
    main()
