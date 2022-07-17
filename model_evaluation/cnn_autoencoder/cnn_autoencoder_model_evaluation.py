import keras.models
import keras.metrics
from typing import final
from training.training_utils import TRAIN_SET_PATH_MEL_SPEC, TEST_SET_PATH_MEL_SPEC, load_dataset
import tensorflow as tf

_EPOCHS_LOAD_CNN: final = 750
_VERSION_LOAD_CNN: final = 1.1
_CNN_NET_PATH: final = f"fitted_autoencoder/cnn/autoencoder_cnn_{_EPOCHS_LOAD_CNN}_epochs_v{_VERSION_LOAD_CNN}"


def main():
    # Load dataset
    train_mel_spec, train_mel_spec_labels = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_mel_spec, test_mel_spec_labels = load_dataset(TEST_SET_PATH_MEL_SPEC)

    model = keras.models.load_model(_CNN_NET_PATH)

    metrics = [tf.keras.metrics.MeanSquaredError(),
               tf.keras.metrics.MeanAbsoluteError(),
               tf.keras.metrics.RootMeanSquaredError()]

    model.compile(metrics=metrics)
    model.evaluate(x=train_mel_spec, y=train_mel_spec)
    model.evaluate(x=test_mel_spec, y=test_mel_spec)


if __name__ == "__main__":
    main()
