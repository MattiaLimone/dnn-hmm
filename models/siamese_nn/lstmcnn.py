import keras
from typing import final, Optional, Union, Any, Iterable
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, TimeDistributed, LeakyReLU, Layer, InputLayer, Softmax, Concatenate, \
    concatenate, RepeatVector
from keras.utils import plot_model
SIAMESE_LSTM_CNN: final = "Siamese_LSTM_Convolutional_Network"
LSTM_BRANCH: final = "Deep_LSTM_Network"
CNN_BRANCH: final = "Deep_Convolutional_Network"
TAIL: final = "LSTM_CNN_Tail"

class LSTMCNN(keras.models.Model):

    def __init__(self, input_shape: tuple[int, ...], lstm_layers: Iterable[Layer], output_dim: int, dense_dim: int,
                 cnn_layers: Iterable[Layer] = None, leaky_relu_alpha: float = 0.05):

        if not input_shape and input_shape[-1] < 1:
            raise ValueError('Feature number must be strictly positive. '
                             f'Received input_shape={input_shape}.')

        '''self._input_shape = input_shape
        self._LSTM_units = LSTM_units'''

        super(LSTMCNN, self).__init__()

        self._lstm_branch = Sequential(name=LSTM_BRANCH)
        self._cnn_branch = Sequential(name=CNN_BRANCH)
        # LSTM BRANCH
        for lstm_layer in lstm_layers:
            # Raise error if one of the given layers is an InputLayer
            if isinstance(lstm_layer, InputLayer):
                raise ValueError("Given layers must not be InputLayer instances")

            self._lstm_branch.add(lstm_layer)
            self._lstm_branch.add(LeakyReLU(alpha=leaky_relu_alpha))
        # CNN BRANCH
        for cnn_layer in cnn_layers:
            if isinstance(cnn_layer, InputLayer):
                raise ValueError("Given layers must not be InputLayer instances")
            self._cnn_branch.add(cnn_layer)

        self._cnn_branch.add(Flatten())
        self._cnn_branch.add(RepeatVector(input_shape[1]))
        # qui vanno gli encoder
        self._lstm_branch.build(input_shape=input_shape)
        self._cnn_branch.build(input_shape=input_shape)
        # NETWORK TAIL
        _merge = TimeDistributed(Concatenate(name="Concatenation_Layer"))(
            [self._lstm_branch.output, self._cnn_branch.output])

        _dense1 = TimeDistributed(Dense(dense_dim, name="Dense_Layer"))(_merge)
        _lrelu1 = TimeDistributed(LeakyReLU(alpha=leaky_relu_alpha))(_dense1)
        _dense2 = TimeDistributed(Dense(output_dim, name="Dense_Layer"))(_lrelu1)
        _output = TimeDistributed(Softmax(name="Output"))(_dense2)

        self.inputs = [self._lstm_branch.input, self._cnn_branch.input]
        self.outputs = _output
        self._tail = Model(_merge, _output, name=TAIL)

        self.build(input_shape=[input_shape, input_shape])

    def call(self, inputs, training=None, mask=None):
        cnn = self._cnn_branch(inputs[0])
        lstm = self._lstm_branch(inputs[1])
        tail = self._tail([cnn, lstm])
        return tail

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested: bool = True,
                show_trainable: bool = False):
        """
        Prints a string summary of the autoencoder network.

            :param line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            :param positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.33, .55, .67, 1.]`.
            :param print_fn: Print function to use. Defaults to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            :param expand_nested: Whether to expand the nested models.
                If not provided, defaults to `True`.
            :param show_trainable: Whether to show if a layer is trainable.
                If not provided, defaults to `False`.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        super(LSTMCNN, self).summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable
        )

    def build_graph(self, inputs):
        """ Plot models that subclass `keras.Model`
        Adapted from https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
        :param inputs: Shape tuple not containing the batch_size
        :return:
        """
        x = keras.Input(shape=inputs)
        return keras.Model(inputs=[x, x], outputs=self.call(x))

    @property
    def lstm_branch(self) -> keras.models.Sequential:
        return self._lstm_branch

    @property
    def cnn_branch(self) -> keras.models.Sequential:
        return self._cnn_branch

    @property
    def tail(self) -> keras.models.Model:
        return self._tail
