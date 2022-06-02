from autoencoder import AutoEncoder
from keras.layers import LSTM


def main():

    lstm0 = LSTM(units=256, activation='relu', return_sequences=True)
    lstm1 = LSTM(units=128, activation='relu', return_sequences=True)
    lstm2 = LSTM(units=64, activation='relu', return_sequences=True)
    layers = [lstm0, lstm1, lstm2]
    model = AutoEncoder(
        39,
        64,
        True,
        True,
        (32, 150, 39),
        *layers,
    )
    # model.build()
    model.summary()


if __name__ == "__main__":
    main()
