from typing import Optional, Any, Union, final
from keras.layers import RNN, BatchNormalization, LayerNormalization, Conv1D, TimeDistributed, Concatenate, Dense, \
    Softmax, RepeatVector, Flatten, Layer, InputLayer, Dropout
from tensorflow.python.keras.layers.pooling import Pooling1D
from keras.models import Model, Sequential
import tensorflow as tf


_RECURRENT_BRANCH_NAME: final = "recurrent_branch"
_CONV_BRANCH_NAME: final = "conv_branch"
_TAIL_NAME: final = "tail"


@tf.keras.utils.register_keras_serializable(package='recconvsnet')
class RecConv1DSiameseNet(Model):
    """
    This class models a recurrent-1D-convolutional double-branched neural network for time-distributed classification
    tasks, composed of a recurrent branch, a convolutional branch and a tail.

    The first one contains recurrent layers (like GRU, LSTM, Conv1DLSTM, Conv2DLSTM, BaseRNN, ...) and other
    time-distributed/utility/normalization layers (TimeDistributed dense layers, BatchNormalization, LayerNormalization,
    RepeatVector layers, ...).

    The convolutional branch contains convolutional and related layers (Conv1D, Pooling1D, ...) and other
    utility/normalization layers (BatchNormalization, LayerNormalization, ...). This branch must either end with a
    TimeDistributed or RepeatVector layer having the same number of timesteps as the LSTM branch.

    Finally, the tail represents the conjunction between the convolutional branch and the recurrent branch, starting
    with a trough-time vector flattening/concatenation (so flattening/concatenation excludes the first two input axes,
    namely batch_size and timesteps). After the concatenation, the input passes through two dense layers with given
    activations and a final dense layer with softmax activation function, which produces the output probabilities.
    """

    def __init__(self,
                 rec_branch_layers: list[RNN, BatchNormalization, RepeatVector, LayerNormalization, Layer],
                 conv_branch_layers: list[Conv1D, BatchNormalization, LayerNormalization, Pooling1D, Dense, Layer],
                 input_shape_rec_branch: tuple[Optional[int], int, ...],
                 input_shape_conv_branch: tuple[Optional[int], int, ...], tail_dense_units: int, output_dim: int,
                 tail_dense_activation='relu', add_repeat_vector_conv_branch: bool = False, dropout_dense=0.0,
                 kernel_regularizer_dense=None, bias_regularizer_dense=None, activity_regularizer_dense=None,
                 kernel_regularizer_softmax=None, bias_regularizer_softmax=None, activity_regularizer_softmax=None,
                 add_double_dense_tail: bool = False):
        """
        Constructor. Instantiates a new RecConv1DSiameseNet model with given parameters.

        :param rec_branch_layers: layers of the recurrent branch of the network, both recurrent (like GRU, LSTM,
            Conv1DLSTM, Conv2DLSTM, BaseRNN, ...) and other time-distributed/utility/normalization layers
            (TimeDistributed dense layers, BatchNormalization, LayerNormalization, RepeatVector layers, ...).
        :param conv_branch_layers: layers of the convolutional branch of the network, both convolutional and related
            layers (like Conv1D, Pooling1D, ...) and other utility/normalization layers (BatchNormalization,
            LayerNormalization, ...).
        :param input_shape_rec_branch: input shape for recurrent branch of the network, must be at least 3-dimensional,
            (batch_size, timesteps, dim1, ..., dimN).
        :param input_shape_conv_branch: input shape for convolutional branch of the network, must be at least
            3-dimensional, (batch_size, dim1, dim2, ..., dimN).
        :param tail_dense_units: number of units of the tail dense layer.
        :param output_dim: number of units/classes of the output layer.
        :param tail_dense_activation: activation function of the tail dense layer (by default, ReLU is used).
        :param add_repeat_vector_conv_branch: whether to add a RepeatVector at the end of the convolutional branch
            (which will have the same number of timesteps as the convolutional branch).
        :param dropout_dense: dropout rate for the dense layer prior to the last layer.
        :param kernel_regularizer_dense: kernel regularizer for the dense layer prior to the last layer.
        :param bias_regularizer_dense: bias regularizer for the dense layer prior to the last layer.
        :param activity_regularizer_dense: activity regularizer for the dense layer prior to the last layer.
        :param kernel_regularizer_dense: kernel regularizer for the last layer.
        :param bias_regularizer_dense: bias regularizer for the last layer.
        :param activity_regularizer_dense: activity regularizer for the last layer.
        :param add_double_dense_tail: whether to add double-dense layer prior to the softmax output layer.
        :raises ValueError: if rec_branch_layers or conv_branch_layers contain any InputLayer, if input shapes are
            incorrect or if given tail_dense_units/units are invalid.
        """

        if len(input_shape_rec_branch) < 3 or len(input_shape_conv_branch) < 3:
            raise ValueError("Input shape for the recurrent and convolutional branches must be at least 3-dimensional")
        if tail_dense_units <= 0:
            raise ValueError("tail_dense_units must be positive")
        if output_dim <= 0:
            raise ValueError("units must be positive")

        super().__init__()

        # Set instance variables
        self.__input_shape_rec_branch = input_shape_rec_branch
        self.__input_shape_conv_branch = input_shape_conv_branch
        self.__tail_dense_units = tail_dense_units
        self.__output_dim = output_dim
        self.__tail_dense_activation = tail_dense_activation
        self.__dropout_dense = dropout_dense
        self.__kernel_regularizer_dense = kernel_regularizer_dense
        self.__bias_regularizer_dense = bias_regularizer_dense
        self.__activity_regularizer_dense = activity_regularizer_dense
        self.__kernel_regularizer_softmax = kernel_regularizer_softmax
        self.__bias_regularizer_softmax = bias_regularizer_softmax
        self.__activity_regularizer_softmax = activity_regularizer_softmax
        self.__add_double_dense_tail = add_double_dense_tail

        # Build recurrent branch
        self.__recurrent_branch = self._build_recurrent_branch(rec_branch_layers)

        # Build convolutional branch
        self.__conv_branch = self._build_conv_branch(conv_branch_layers, add_repeat_vector_conv_branch)

        # Build tail
        self.__tail = self._build_tail()

        # Build the model
        self.build(input_shape=[input_shape_rec_branch, input_shape_conv_branch])

    def _build_recurrent_branch(self, rec_branch_layers: list) -> Sequential:
        """
        Builds the recurrent branch of the network.

        :param rec_branch_layers: layers of the recurrent branch of the network, both recurrent (like GRU, LSTM,
            Conv1DLSTM, Conv2DLSTM, BaseRNN, ...) and other time-distributed/utility/normalization layers
            (TimeDistributed dense layers, BatchNormalization, LayerNormalization, RepeatVector layers, ...).
        :return: a Sequential model representing the recurrent branch.
        :raises ValueError: if rec_branch_layers contains any InputLayer.
        """
        recurrent_branch = Sequential(name=_RECURRENT_BRANCH_NAME)

        # Add all the given layers to the recurrent branch Sequential model
        for rec_branch_layer in rec_branch_layers:
            if isinstance(rec_branch_layer, InputLayer):
                raise ValueError("Recurrent branch cannot contain InputLayer(s).")
            recurrent_branch.add(rec_branch_layer)

        # Build the recurrent branch
        recurrent_branch.build(input_shape=self.__input_shape_rec_branch)
        return recurrent_branch

    def _build_conv_branch(self, conv_branch_layers: list, add_repeat_vector_conv_branch: bool) -> Sequential:
        """
        Builds the convolutional branch of the network.

        :param conv_branch_layers: layers of the convolutional branch of the network, both convolutional and related
            layers (like Conv1D, Pooling1D, ...) and other utility/normalization layers (BatchNormalization,
            LayerNormalization, ...).
        :param add_repeat_vector_conv_branch: whether to add a RepeatVector at the end of the convolutional branch
            (which will have the same number of timesteps as the convolutional branch).
        :return: a Sequential model representing the convolutional branch of the network.
        :raises ValueError: if conv_branch_layers contains any InputLayer.
        """
        conv_branch = Sequential(name=_CONV_BRANCH_NAME)

        # Add all the given layers to the convolutional branch Sequential model
        for conv_branch_layer in conv_branch_layers:
            if isinstance(conv_branch_layer, InputLayer):
                raise ValueError("Recurrent branch cannot contain InputLayer(s).")
            conv_branch.add(conv_branch_layer)

        # If required, add flatten and repeat vector layers to allow concatenation with recurrent branch frames overtime
        timesteps = self.__recurrent_branch.output_shape[1]
        if add_repeat_vector_conv_branch:
            conv_branch.add(Flatten(name="conv_branch_flatten"))
            conv_branch.add(RepeatVector(n=timesteps, name="conv_branch_repeat_vector"))

        # Build the convolutional branch
        conv_branch.build(input_shape=self.__input_shape_conv_branch)

        # Check if the number of timesteps from the convolutional branch is equal to one from the recurrent branch
        if conv_branch.output_shape[1] != timesteps:
            raise ValueError(
                "The number of timesteps of the convolutional branch must be equal to the one of the recurrent branch"
            )
        return conv_branch

    def _build_tail(self) -> Model:
        """
        Builds the tail of the double-branched network.

        :return: a keras Model representing the tail of the double-branched network.
        """
        merge = TimeDistributed(
            Concatenate(),
            name="tail_concatenate_layer"
        )([self.__recurrent_branch.output, self.__conv_branch.output])
        dense0 = TimeDistributed(
            Dense(
                units=self.__tail_dense_units,
                activation=self.__tail_dense_activation,
                kernel_regularizer=self.__kernel_regularizer_dense,
                bias_regularizer=self.__bias_regularizer_dense,
                activity_regularizer=self.__activity_regularizer_dense,
            ),
            name="tail_dense0"
        )(merge)
        if self.__dropout_dense > 0:
            dense0 = TimeDistributed(
                Dropout(rate=self.__dropout_dense),
                name="tail_dropout_dense_0"
            )(dense0)

        dense1 = dense0
        if self.__add_double_dense_tail:
            dense1 = TimeDistributed(
                Dense(
                    units=self.__tail_dense_units,
                    activation=self.__tail_dense_activation,
                    kernel_regularizer=self.__kernel_regularizer_dense,
                    bias_regularizer=self.__bias_regularizer_dense,
                    activity_regularizer=self.__activity_regularizer_dense,
                ),
                name="tail_dense1"
            )(dense0)
            if self.__dropout_dense > 0:
                dense1 = TimeDistributed(
                    Dropout(rate=self.__dropout_dense),
                    name="tail_dropout_dense_1"
                )(dense1)

        dense2 = TimeDistributed(
            Dense(
                units=self.__output_dim,
                kernel_regularizer=self.__kernel_regularizer_softmax,
                bias_regularizer=self.__bias_regularizer_softmax,
                activity_regularizer=self.__activity_regularizer_softmax
            ),
            name="tail_output_dense"
        )(dense1)
        output = TimeDistributed(
            Softmax(),
            name="tail_output_softmax"
        )(dense2)
        tail = Model(
            inputs=[self.__recurrent_branch.output, self.__conv_branch.output],
            outputs=output,
            name=_TAIL_NAME
        )
        return tail

    @property
    def input_shape_rec_branch(self) -> tuple[Optional[int], int, ...]:
        """
        Retrieves the input shape of the recurrent branch of the network.

        :return: a tuple representing the input shape of the recurrent branch of the network.
        """
        return self.__input_shape_rec_branch

    @property
    def input_shape_conv_branch(self) -> tuple[Optional[int], int, ...]:
        """
        Retrieves the input shape of the convolutional branch of the network.

        :return: a tuple representing the input shape of the convolutional branch of the network.
        """
        return self.__input_shape_conv_branch

    @property
    def tail_dense_units(self) -> int:
        """
        Retrieves the unit number of units of the tail dense layers.

        :return: an integer representing the unit number of the tail dense layers.
        """
        return self.__tail_dense_units

    @property
    def output_dim(self) -> int:
        """
        Retrieves the output dimension.

        :return: an integer representing the output dimensionality (i.e. number of output classes).
        """
        return self.__output_dim

    @property
    def tail_dense_activation(self):
        """
        Retrieves the activation function of the tail dense layers.

        :return: the activation function of the tail dense layers.
        """
        return self.__tail_dense_activation

    @property
    def dropout_dense(self) -> float:
        """
        Retrieves the dropout rate applied to the dense layers of the tail.

        :return: a float representing the dropout rate applied to the dense layers of the tail.
        """
        return self.__dropout_dense

    @property
    def kernel_regularizer_dense(self):
        """
        Retrieves the kernel regularization function applied to the tail dense layers (except the last, softmax one).

        :return: the kernel regularization function applied to the tail dense layers (except the last, softmax one).
        """
        return self.__kernel_regularizer_dense

    @property
    def bias_regularizer_dense(self):
        """
        Retrieves the bias regularization function applied to the tail dense layers (except the last, softmax one).

        :return: the bias regularization function applied to the tail dense layers (except the last, softmax one).
        """
        return self.__bias_regularizer_dense

    @property
    def activity_regularizer_dense(self):
        """
        Retrieves the activity regularization function applied to the tail dense layers (except the last, softmax one).

        :return: the activity regularization function applied to the tail dense layers (except the last, softmax one).
        """
        return self.__activity_regularizer_dense

    @property
    def kernel_regularizer_softmax(self):
        """
        Retrieves the kernel regularization function applied to the last softmax dense layer.

        :return: the kernel regularization function applied to the last softmax dense layer.
        """
        return self.__kernel_regularizer_softmax

    @property
    def bias_regularizer_softmax(self):
        """
        Retrieves the bias regularization function applied to the last softmax dense layer.

        :return: the bias regularization function applied to the last softmax dense layer.
        """
        return self.__bias_regularizer_softmax

    @property
    def activity_regularizer_softmax(self):
        """
        Retrieves the activity regularization function applied to the last softmax dense layer.

        :return: the activity regularization function applied to the last softmax dense layer.
        """
        return self.__activity_regularizer_softmax

    @property
    def rec_branch_layers(self) -> list[Layer]:
        """
        Retrieves the layers of the recurrent branch of the network.

        :return: a list containing the layers of the recurrent branch.
        """
        return self.__recurrent_branch.layers

    @property
    def conv_branch_layers(self) -> list[Layer]:
        """
        Retrieves the layers of the convolutional branch of the network.

        :return: a list containing the layers of the convolutional branch.
        """
        return self.__conv_branch.layers

    def call(self, inputs, training=None, mask=None):
        rec_branch_input, conv_branch_input = inputs
        rec_output = self.__recurrent_branch(rec_branch_input)
        conv_output = self.__conv_branch(conv_branch_input)
        return self.__tail([rec_output, conv_output])

    def get_config(self) -> dict[str, Union[None, list[Optional[dict[str, Any]]], tuple, int]]:
        config_dict = {
            "rec_branch_layers": self.rec_branch_layers,
            "conv_branch_layers": self.conv_branch_layers,
            "input_shape_rec_branch": self.input_shape_rec_branch,
            "input_shape_conv_branch": self.input_shape_conv_branch,
            "tail_dense_units": self.tail_dense_units,
            "output_dim": self.output_dim,
            "tail_dense_activation": self.tail_dense_activation,
            "add_repeat_vector_conv_branch": False,
            "dropout_dense": self.dropout_dense,
            "kernel_regularizer_dense": self.kernel_regularizer_dense,
            "bias_regularizer_dense": self.bias_regularizer_dense,
            "activity_regularizer_dense": self.activity_regularizer_dense,
            "kernel_regularizer_softmax": self.kernel_regularizer_softmax,
            "bias_regularizer_softmax": self.bias_regularizer_softmax,
            "activity_regularizer_softmax": self.activity_regularizer_softmax,
            "add_double_dense_tail": self.__add_double_dense_tail
        }
        return config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested: bool = True,
                show_trainable: bool = False):
        """
        Prints a string summary of the recurrent-1D-convolutional siamese neural network.

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
        super(RecConv1DSiameseNet, self).summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable
        )
