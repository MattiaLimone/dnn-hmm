from autoencoder import AutoEncoder
from keras.layers import LSTM


def main():

    lstm0 = LSTM(units=256, activation='relu', return_sequences=True)
    lstm1 = LSTM(units=128, activation='relu', return_sequences=True)
    lstm2 = LSTM(name='Bottleneck', units=64, activation='relu', return_sequences=True)
    lstm_decoder0 = LSTM(name="lstm_decoder", units=128, activation='relu', return_sequences=True)
    lstm_decoder1 = LSTM(name="lstm_decoder1", units=256,  activation='relu', return_sequences=True)
    encoder_layers = [lstm0, lstm1]
    decoder_layers = [lstm_decoder0, lstm_decoder1]
    model = AutoEncoder(
        n_features=39,
        encoder_layers=encoder_layers,
        bottleneck=lstm2,
        decoder_layers=decoder_layers,
        outputs_sequences=True,
        input_shape=(32, 150, 39),
    )
    # model.build()
    model.summary()

    # TODO: add model training


if __name__ == "__main__":
    main()
