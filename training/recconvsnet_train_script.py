from typing import final
import keras.models
import keras.losses
from keras.optimizer_v2.adadelta import Adadelta
from matplotlib import pyplot
from models.recconvsnet.recconvsnet import RecConv1DSiameseNet
from models.autoencoder.autoencoder import ENCODER_MODEL_NAME
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    TEST_SET_PATH_MEL_SPEC, load_dataset, get_label_number


_EPOCHS_LOAD_CONV: final = 1050
_EPOCHS_LOAD_REC: final = 750
_CONV_AUTOENCODER_PATH: final = f"fitted_autoencoder/cnn/autoencoder_cnn_{_EPOCHS_LOAD_CONV}_epochs"
_REC_AUTOENCODER_PATH: final = f"fitted_autoencoder/lstm/autoecoder_lstm_{_EPOCHS_LOAD_REC}_epochs"


def main():
    # Load dataset and labels
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    train_mel_spec, train_mel_spec_labels = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_mel_spec, test_mel_spec_labels = load_dataset(TEST_SET_PATH_MEL_SPEC)
    total_state_number = get_label_number(train_mfccs_labels)

    # Load saved models
    conv_autoencoder = keras.models.load_model(_CONV_AUTOENCODER_PATH)
    rec_autoencoder = keras.models.load_model(_REC_AUTOENCODER_PATH)

    # Get input shapes
    input_shape_conv_branch = conv_autoencoder.input_shape
    input_shape_rec_branch = rec_autoencoder.input_shape
    timesteps = input_shape_rec_branch[1]

    # Get recurrent and convolutional encoder layers
    conv_branch = conv_autoencoder.get_layer(ENCODER_MODEL_NAME).layers
    rec_branch = rec_autoencoder.get_layer(ENCODER_MODEL_NAME).layers

    # Set model parameters
    tail_dense_units = 1024

    # Set model training parameters
    epochs = 500
    batch_size = 250
    loss = keras.losses.CategoricalCrossentropy(
        from_logits=False,
        name="categorical_cross_entropy_loss"
    )
    optimizer = Adadelta(
        learning_rate=1,
        rho=0.95,
        epsilon=1e-7,
        name='adadelta_optimizer'
    )
    callbacks = None  # This can be replaced with custom early stopping callbacks
    version = 0.1  # For easy saving of multiple model versions

    # Instantiate the model and compile it
    model = RecConv1DSiameseNet(
        rec_branch_layers=rec_branch,
        conv_branch_layers=conv_branch,
        input_shape_rec_branch=input_shape_rec_branch,
        input_shape_conv_branch=input_shape_conv_branch,
        tail_dense_units=tail_dense_units,
        output_dim=total_state_number,
        tail_dense_activation='relu',
        timesteps_repeat_vector_conv_branch=timesteps
    )
    model.compile(optimizer=optimizer, loss=loss,)
    model.summary(expand_nested=True)

    # Train the model
    history = model.fit(
        x=[train_mfccs, train_mel_spec],
        y=train_mfccs_labels,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=([test_mfccs, test_mel_spec], test_mfccs_labels),
    )

    # Plot results
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # Save the model to file
    model.save(f'fitted_recconvsnet/recconvsnet_{epochs}_epochs_v{version}')


if __name__ == "__main__":
    main()
