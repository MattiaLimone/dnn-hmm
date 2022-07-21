from keras import Sequential, regularizers
import keras.models
import keras.metrics
import tensorflow as tf
from typing import final
from models.recconvsnet.recconvsnet import RecConv1DSiameseNet, load_recconvsnet
from preprocessing.constants import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, AUDIO_PER_SPEAKER, AUDIO_DATAFRAME_KEY,\
    STATE_PROB_KEY, N_STATES_MFCCS
from preprocessing.file_utils import generate_or_load_speaker_ordered_dict, load_speakers_acoustic_models
from training.training_utils import TRAIN_SET_PATH_MFCCS, TEST_SET_PATH_MFCCS, TRAIN_SET_PATH_MEL_SPEC, \
    TEST_SET_PATH_MEL_SPEC, load_dataset, speaker_n_states_in_top_k_accuracy_mfccs, one_hot_labels_to_integer_labels, \
    sparse_top_k_categorical_speaker_accuracy_mfccs, get_label_number

_EPOCHS_LOAD_RECCONV: final = 600
_VERSION_LOAD_RECCONV: final = 1.1
_RECCONV_NET_PATH: final = f"fitted_recconvsnet/recconvsnet_{_EPOCHS_LOAD_RECCONV}_epochs_v{_VERSION_LOAD_RECCONV}"


def main():
    # Load dataset
    train_mfccs, train_mfccs_labels = load_dataset(TRAIN_SET_PATH_MFCCS)
    test_mfccs, test_mfccs_labels = load_dataset(TEST_SET_PATH_MFCCS)
    train_mel_spec, train_mel_spec_labels = load_dataset(TRAIN_SET_PATH_MEL_SPEC)
    test_mel_spec, test_mel_spec_labels = load_dataset(TEST_SET_PATH_MEL_SPEC)

    labels_train = one_hot_labels_to_integer_labels(train_mfccs_labels)
    labels_test = one_hot_labels_to_integer_labels(test_mfccs_labels)
    total_state_number = get_label_number(train_mfccs_labels)

    validation_limit = int(len(test_mfccs)/2)
    model = load_recconvsnet(path=_RECCONV_NET_PATH, custom_objects={
        "speaker_n_states_in_top_k_accuracy_mfccs": speaker_n_states_in_top_k_accuracy_mfccs,
        "sparse_top_k_categorical_speaker_accuracy_mfccs": sparse_top_k_categorical_speaker_accuracy_mfccs
    })
    '''
    model = keras.models.load_model(_RECCONV_NET_PATH, custom_objects={"RecConv1DSiameseNet": RecConv1DSiameseNet})
    
    model_copy = RecConv1DSiameseNet(
        rec_branch_layers=model.get_layer("recurrent_branch").layers,
        conv_branch_layers=model.get_layer("conv_branch").layers,
        input_shape_rec_branch=(None,) + train_mfccs.shape[1:],
        input_shape_conv_branch=(None, ) + train_mel_spec.shape[1:],
        tail_dense_units=total_state_number,
        output_dim=total_state_number,
        tail_dense_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        add_repeat_vector_conv_branch=False,
        kernel_regularizer_dense=regularizers.L1(1e-4),
        activity_regularizer_softmax=regularizers.L1(1e-4),
        dropout_dense=dropout_rate,
        add_double_dense_tail=False
    )
    model_copy.set_weights(model.get_weights())
    model = model_copy
    '''


    #model_copy = RecConv1DSiameseNet.from_config(model.get_config())
    #model_copy.set_weights(model.get_weights())
    #model = model_copy

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy', dtype=None),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=N_STATES_MFCCS, name='Top-K accuracy', dtype=None),
        speaker_n_states_in_top_k_accuracy_mfccs,
        sparse_top_k_categorical_speaker_accuracy_mfccs
    ]

    model.compile(metrics=metrics)
    model.predict([train_mfccs[:1], train_mel_spec[:1]])

    model.evaluate(x=[train_mfccs, train_mel_spec], y=labels_train)
    model.evaluate(
        x=[test_mfccs[validation_limit:], test_mel_spec[validation_limit:]],
        y=labels_test[validation_limit:]
    )



if __name__ == "__main__":
    main()
