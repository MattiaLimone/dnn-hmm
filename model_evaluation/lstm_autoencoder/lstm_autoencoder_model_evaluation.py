import keras.models
import keras.metrics
from typing import final
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, load_dataset, coeff_determination
import tensorflow as tf

_EPOCHS_LOAD_LSTM: final = 800
_VERSION_LOAD_LSTM: final = 1.1
_LSTM_NET_PATH: final = f"fitted_autoencoder/lstm/autoencoder_lstm_{_EPOCHS_LOAD_LSTM}_epochs_v{_VERSION_LOAD_LSTM}"


def main():
    # Load dataset
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)

    model =  keras.models.load_model(_LSTM_NET_PATH)

    metrics = [coeff_determination,
               tf.keras.metrics.MeanAbsoluteError(),
               tf.keras.metrics.RootMeanSquaredError()]

    model.compile(metrics=metrics)
    model.evaluate(x=train_mfccs, y=train_mfccs)
    model.evaluate(x=test_mfccs, y=test_mfccs)


if __name__ == "__main__":
    main()
