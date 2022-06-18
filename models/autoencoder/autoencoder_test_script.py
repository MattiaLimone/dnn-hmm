from autoencoder import AutoEncoder
from keras.layers import LSTM


def main():

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
        input_shape=(10, 150, 39),
    )
    # model.build()
    model.summary()

    # TODO: add model training


if __name__ == "__main__":
    main()
