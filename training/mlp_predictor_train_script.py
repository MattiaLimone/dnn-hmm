from typing import final
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizer_v2.adadelta import Adadelta
import keras.regularizers as regularizers
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Input, Dropout, \
    TimeDistributed
from models.autoencoder.autoencoder import ENCODER_MODEL_NAME
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS,load_dataset, get_label_number, \
    one_hot_labels_to_integer_labels, \
    sparse_categorical_speaker_accuracy_mfccs, sparse_top_k_categorical_speaker_accuracy_mfccs, \
    speaker_n_states_in_top_k_accuracy_mfccs


_EPOCHS_LOAD_REC: final = 1000
_VERSION_LOAD_REC: final = 1.0
_REC_AUTOENC_PATH: final = f"fitted_autoencoder/lstm/autoencoder_lstm_{_EPOCHS_LOAD_REC}_epochs_v{_VERSION_LOAD_REC}"


def main():
    # Load dataset and labels
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    total_state_number = get_label_number(train_mfccs_labels)

    # Get input shapes
    input_shape_rec_branch = (None, ) + train_mfccs.shape[1:]

    # Set model parameters
    tail_dense_units = 512

    # Set model training parameters
    epochs = 1
    batch_size = 100

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        name='sparse_categorical_crossentropy'
    )
    optimizer = Adadelta(
        learning_rate=1,
        rho=0.95,
        epsilon=1e-7,
        name='adadelta_optimizer'
    )
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="Accuracy", dtype=None),
               tf.keras.metrics.SparseTopKCategoricalAccuracy(k=8, name="TopK Accuracy", dtype=None),
               sparse_top_k_categorical_speaker_accuracy_mfccs,
               sparse_categorical_speaker_accuracy_mfccs,
               speaker_n_states_in_top_k_accuracy_mfccs]
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, min_delta=0.001, restore_best_weights=True)
    ]

    version = 0.4  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    model = Sequential()
    model.add(Input(shape=(train_mfccs.shape[1:])))
    model.add(Dense(units=tail_dense_units,
                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                    kernel_regularizer=regularizers.L1(1e-5)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=tail_dense_units,
                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                    kernel_regularizer=regularizers.L1(1e-5)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(total_state_number,
                    activation='softmax',
                    activity_regularizer=regularizers.L1(1e-5)))
    model.build(input_shape_rec_branch)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(expand_nested=True)

    # Convert one-hot encoded labels to integer labels
    labels_train = one_hot_labels_to_integer_labels(train_mfccs_labels)
    labels_test = one_hot_labels_to_integer_labels(test_mfccs_labels)

    # Train the model
    history = model.fit(
        x=train_mfccs,
        y=labels_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(test_mfccs, labels_test)
    )
    # Save the model to file
    model.save(f'fitted_mlp_predictor/mlp_predictor_{epochs}_epochs_v{version}')

    # Plot results loss
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
