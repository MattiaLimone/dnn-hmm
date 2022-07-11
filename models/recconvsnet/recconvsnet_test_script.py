from recconvsnet import RecConv1DSiameseNet
from keras.layers import LSTM, Conv1D, AvgPool1D, BatchNormalization
from models.autoencoder.autoencoder import FlattenDenseLayer


def main():
    timesteps = 300
    input_shape_conv_branch = (None, timesteps, 128)
    input_shape_lstm_branch = (None, timesteps, 39)
    output_dim = 5 * 630

    lstm0 = LSTM(name="lstm0", units=512, activation='tanh', return_sequences=True)
    lstm1 = LSTM(name="lstm1", units=256, activation='tanh', return_sequences=True)
    lstm2 = LSTM(name='Bottleneck', units=128, activation='tanh', return_sequences=True)
    bnlstm = BatchNormalization(name="lstm_branch_batch_normalization")

    conv0 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name="conv0")
    bn0 = BatchNormalization(name="conv_branch_batch_normalization0")
    avg_pool0 = AvgPool1D(pool_size=2, name="avg_pool0", padding='same')
    conv1 = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', name="conv1")
    bn1 = BatchNormalization(name="conv_branch_batch_normalization1")
    avg_pool1 = AvgPool1D(pool_size=2, name="avg_pool1", padding='same')
    conv2 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', name="conv2")
    bn2 = BatchNormalization(name="conv_branch_batch_normalization2")
    avg_pool2 = AvgPool1D(pool_size=2, name="avg_pool2", padding='same')
    conv_branch_flatten = FlattenDenseLayer(units=512, name="conv_branch_flatten-dense")

    model = RecConv1DSiameseNet(
        rec_branch_layers=[lstm0, lstm1, lstm2, bnlstm],
        conv_branch_layers=[conv0, bn0, avg_pool0, conv1, bn1, avg_pool1, conv2, bn2, avg_pool2, conv_branch_flatten],
        input_shape_rec_branch=input_shape_lstm_branch,
        input_shape_conv_branch=input_shape_conv_branch,
        tail_dense_units=1024,
        output_dim=output_dim,
        tail_dense_activation='relu',
        add_repeat_vector_conv_branch=True
    )
    model.summary()


if __name__ == "__main__":
    main()
