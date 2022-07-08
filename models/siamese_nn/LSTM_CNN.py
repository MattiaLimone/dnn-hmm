import keras
from typing import final, Optional, Union, Any, Iterable
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv1D, MaxPool1D, AvgPool1D, Flatten, Concatenate, LeakyReLU, Input, Softmax, \
    BatchNormalization, RepeatVector, TimeDistributed
from keras.utils import plot_model

def build_LSTM_CNN(lstm_input_shape: tuple[int, ...], cnn_input_shape: tuple[int, ...], lstm_units: int = 128,
                   leaky_relu_alpha: float = 0.05, output_units: int = 100):
    #LSTM_BRANCH
    lstm_input = Input(name="LSTM_Input", shape=lstm_input_shape[1:])
    lstm1 = LSTM(name="LSTM_Layer_1", units=lstm_units, return_sequences=True)(lstm_input)
    lrelu1 = LeakyReLU(name="LeakyRelu_Activation_1", alpha=leaky_relu_alpha)(lstm1)
    lstm2 = LSTM(name="LSTM_Layer_2", units=lstm_units, return_sequences=True)(lrelu1)
    lrelu2 = LeakyReLU(name="LeakyRelu_Activation_2", alpha=leaky_relu_alpha)(lstm2)
    lstm3 = LSTM(name="LSTM_Layer_3", units=lstm_units, return_sequences=True)(lrelu2)
    lrelu3 = LeakyReLU(name="LeakyRelu_Activation_3", alpha=leaky_relu_alpha)(lstm3)
    #flatten1 = Flatten(name="LSTM_Flatten")(lrelu3)
    #CNN_BRANCH
    cnn_input = Input(name="CNN_Input", shape=cnn_input_shape[1:])
    cnn1 = Conv1D(32, 1, activation='relu', name="Convolutional_Layer_1")(cnn_input)
    bn1 = BatchNormalization(name="Batch_Normalization_1")(cnn1)
    avp1 = AvgPool1D(name="Average_Pool_1")(bn1)
    cnn2 = Conv1D(32, 3, activation='relu', name="Convolutional_Layer_2")(avp1)
    bn2 = BatchNormalization(name="Batch_Normalization_2")(cnn2)
    avp2 = AvgPool1D(name="Average_Pool_2")(bn2)
    cnn3 = Conv1D(32, 1, activation='relu', name="Convolutional_Layer_3")(avp2)
    avp3 = AvgPool1D(name="Average_Pool_3")(cnn3)
    #flatten2 = Flatten(name="CNN_Flatten")(avp3)
    #reapeatvector = RepeatVector(cnn_input_shape[1])(avp3)
    #TAIL_BRANCH
    merge = TimeDistributed(Concatenate(name="Concatenation_Layer"))([avp3, lrelu3])
    dense1 = TimeDistributed(Dense(1024, name="Dense_Layer"))(merge)
    lrelu4 = TimeDistributed(LeakyReLU(alpha=leaky_relu_alpha))(dense1)
    output = TimeDistributed(Softmax(name="Output"))(lrelu4)
    #RETURN MODEL
    _model = Model([lstm_input, cnn_input], output)

    return _model


n_features = 100
n_timesteps = 160
batch_size = 10
shape_input1 = (batch_size, n_timesteps, n_features)
shape_input2 = (batch_size, n_timesteps, n_features*2)

model = build_LSTM_CNN(lstm_input_shape=shape_input1,
                       cnn_input_shape=shape_input2,
                       lstm_units=100,
                       output_units=100)
model.summary()
plot_model(model, show_shapes=True)