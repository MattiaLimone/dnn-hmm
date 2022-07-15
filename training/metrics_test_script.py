import tensorflow as tf
from preprocessing.constants import N_STATES_MFCCS
from training_utils import speaker_n_states_in_top_k_accuracy_mfccs, sparse_categorical_speaker_accuracy_mfccs, \
    sparse_top_k_categorical_speaker_accuracy_mfccs


def main():
    y_pred = tf.convert_to_tensor([
            [[0, 0, 0.1, 0.05, 0.05, 0.05, 0.01, 0.02, 0, 0, 0.18, 0.05, 0, 0.05, 0, 0.54],
             [0, 0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.02, 0, 0, 0, 0.5, 0.3, 0.005, 0.005, 0.21],
             [0, 0, 0.5, 0.07, 0.03, 0.19, 0.01, 0.2, 0, 0.05, 0, 0.01, 0.3, 0.005, 0.005, 0.13]],
            [[0, 0, 0.1, 0.05, 0.2, 0, 0, 0.05, 0, 0, 0.05, 0.4, 0.2, 0.05, 0.05, 0.14],
             [0, 0, 0.15, 0.2, 0.25, 0.01, 0.15, 0.1, 0.05, 0.05, 0, 0.05, 0.05, 0, 0, 0.14],
             [0.04, 0.06, 0.03, 0.25, 0.22, 0.3, 0.03, 0.08, 0, 0.07, 0, 0, 0.03, 0.05, 0.05, 0.14]]
    ])
    y_true = tf.convert_to_tensor([[0, 3, 5], [11, 8, 9]])

    print(f"y_pred: {str(y_pred)}")
    print(f"y_pred: {str(y_true)}")
    m0 = sparse_top_k_categorical_speaker_accuracy_mfccs(y_true, y_pred, k=2)
    m1 = sparse_categorical_speaker_accuracy_mfccs(y_true, y_pred)
    m2 = speaker_n_states_in_top_k_accuracy_mfccs(y_true, y_pred)

    print("Metrics: ")
    print(m0)
    print(m1)
    print(m2)
    print(f"Top {N_STATES_MFCCS} max in each row: {str(tf.math.top_k(y_pred, k=N_STATES_MFCCS))}")


if __name__ == "__main__":
    main()
