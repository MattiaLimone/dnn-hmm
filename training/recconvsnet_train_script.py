from typing import final
import tensorflow as tf
import keras.models
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
import keras.regularizers as regularizers
from keras.layers import Dropout
from matplotlib import pyplot
from models.recconvsnet.recconvsnet import RecConv1DSiameseNet
from models.autoencoder.autoencoder import ENCODER_MODEL_NAME
from preprocessing.constants import N_STATES_MFCCS
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    TEST_SET_PATH_MEL_SPEC, load_dataset, get_label_number, one_hot_labels_to_integer_labels, \
    sparse_top_k_categorical_speaker_accuracy_mfccs, speaker_n_states_in_top_k_accuracy_mfccs


_EPOCHS_LOAD_CONV: final = 4000
_EPOCHS_LOAD_REC: final = 800
_VERSION_LOAD_CONV: final = 1.5
_VERSION_LOAD_REC: final = 1.1
_CONV_AUTOENC_PATH: final = f"fitted_autoencoder/cnn/autoencoder_cnn_{_EPOCHS_LOAD_CONV}_epochs_v{_VERSION_LOAD_CONV}"
_REC_AUTOENC_PATH: final = f"fitted_autoencoder/lstm/autoencoder_lstm_{_EPOCHS_LOAD_REC}_epochs_v{_VERSION_LOAD_REC}"
_EPOCHS_LOAD_RECCONV: final = 1000
_VERSION_LOAD_RECCONV: final = 1.0
_RECCONV_NET_PATH: final = f"fitted_autoencoder/cnn/autoencoder_cnn_{_EPOCHS_LOAD_RECCONV}_epochs_" \
                           f"v{_VERSION_LOAD_RECCONV}"


def main():
    # Load dataset and labels
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    train_mel_spec, train_mel_spec_labels = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_mel_spec, test_mel_spec_labels = load_dataset(TEST_SET_PATH_MEL_SPEC)
    total_state_number = get_label_number(train_mfccs_labels)

    # Load saved models
    conv_autoencoder = tf.keras.models.load_model(_CONV_AUTOENC_PATH)
    rec_autoencoder = tf.keras.models.load_model(_REC_AUTOENC_PATH)

    # Get input shapes
    input_shape_conv_branch = (None,) + train_mel_spec.shape[1:]
    input_shape_rec_branch = (None,) + train_mfccs.shape[1:]

    # Get recurrent and convolutional encoder layers
    conv_branch = conv_autoencoder.get_layer(ENCODER_MODEL_NAME).layers
    rec_branch = rec_autoencoder.get_layer(ENCODER_MODEL_NAME).layers

    # Add additional layers
    dropout_rate_conv = 0.5
    dropout_rate_rec = 0.7
    dropout_rate = 0.5
    rec_drop_layer_numb = 5
    conv_drop_layer_numb = 0
    final_dropout_layer = True

    # Add Dropout layers backwards to LSTM Branch
    for i in range(rec_drop_layer_numb, 0, -1):
        rec_branch.insert(i, Dropout(rate=dropout_rate_conv))
    # Add Dropout layers backwards to Conv Branch
    for i in range(conv_drop_layer_numb, 0, -1):
        conv_branch.insert(i, Dropout(rate=dropout_rate_rec))
    # Add Dropout before concatenate layer
    if final_dropout_layer:
        conv_branch.append(Dropout(rate=dropout_rate_conv))
        rec_branch.append(Dropout(rate=dropout_rate_rec))

    # Set model parameters
    tail_dense_units = total_state_number

    # Set model training parameters
    epochs = 600

    batch_size = 200
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
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, min_delta=0.001, restore_best_weights=True)
    ]
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy', dtype=None),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=N_STATES_MFCCS, name='Top-K accuracy', dtype=None),
        #sparse_top_k_categorical_speaker_accuracy_mfccs,
        #speaker_n_states_in_top_k_accuracy_mfccs
    ]
    version = 1.1  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    retraining = int(input("Insert 0 for training and 1 for retraining: "))
    if retraining == 0:
        model = RecConv1DSiameseNet(
            rec_branch_layers=rec_branch,
            conv_branch_layers=conv_branch,
            input_shape_rec_branch=input_shape_rec_branch,
            input_shape_conv_branch=input_shape_conv_branch,
            tail_dense_units=tail_dense_units,
            output_dim=total_state_number,
            tail_dense_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            add_repeat_vector_conv_branch=True,
            kernel_regularizer_dense=regularizers.L1(1e-4),
            activity_regularizer_softmax=regularizers.L1(1e-4),
            dropout_dense=dropout_rate,
            add_double_dense_tail=False
        )
    else:
        model = keras.models.load_model(_RECCONV_NET_PATH)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(expand_nested=True)

    # Convert one-hot encoded labels to integer labels
    labels_train = one_hot_labels_to_integer_labels(train_mfccs_labels)
    labels_test = one_hot_labels_to_integer_labels(test_mfccs_labels)

    # Train the model
    history = model.fit(
        x=[train_mfccs, train_mel_spec],
        y=labels_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=([test_mfccs, test_mel_spec], labels_test)
    )
    # Save the autoencoder to file
    if retraining == 0:
        model.save(f'fitted_recconvsnet/recconvsnet_{epochs}_epochs_v{version}')
    else:
        model.save(f'fitted_recconvsnet/recconvsnet_{epochs + _EPOCHS_LOAD_CONV}_epochs_v{version}')

    # Plot results loss
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    '''
    # Plot results accuracy
    pyplot.plot(history.history['sparse_categorical_accuracy'], label='train sparse categorical accuracy')
    pyplot.plot(history.history['val_sparse_categorical_accuracy'], label='test parse categorical accuracy')
    pyplot.legend()
    pyplot.show()
    '''


if __name__ == "__main__":
    main()
