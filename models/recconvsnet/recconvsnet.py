from typing import Optional, Any, Union, final
from keras.layers import RNN, BatchNormalization, LayerNormalization, Conv1D, TimeDistributed, Concatenate, Dense, \
    Softmax, RepeatVector, Flatten, Layer, InputLayer, Dropout
from keras.layers.pooling import Pooling1D
from keras.models import Model, Sequential


_RECURRENT_BRANCH_NAME: final = "recurrent_branch"
_CONV_BRANCH_NAME: final = "conv_branch"
_TAIL_NAME: final = "tail"


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
    namely batch_size and timesteps). After the concatenation, the input passes through two dense layers and a final
    softmax activation function, which produces the output probabilities.
    """

    def __init__(self,
                 rec_branch_layers: list[RNN, BatchNormalization, RepeatVector, LayerNormalization, Layer],
                 conv_branch_layers: list[Conv1D, BatchNormalization, LayerNormalization, Pooling1D, Dense, Layer],
                 input_shape_rec_branch: tuple[Optional[int], int, ...],
                 input_shape_conv_branch: tuple[Optional[int], int, ...], tail_dense_units: int, output_dim: int,
                 tail_dense_activation='relu', add_repeat_vector_conv_branch: bool = False, dropout_dense=0.0,
                 kernel_regularizer_dense=None, bias_regularizer_dense=None, activity_regularizer_dense=None,
                 kernel_regularizer_softmax=None, bias_regularizer_softmax=None, activity_regularizer_softmax=None):
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
        :raises ValueError: if rec_branch_layers or conv_branch_layers contain any InputLayer, if input shapes are
            incorrect or if given tail_dense_units/output_dim are invalid.
        """

        if len(input_shape_rec_branch) < 3 or len(input_shape_conv_branch) < 3:
            raise ValueError("Input shape for the recurrent and convolutional branches must be at least 3-dimensional")
        if tail_dense_units <= 0:
            raise ValueError("tail_dense_units must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        super().__init__()

        # Set instance variables
        self.__input_shape_rec_branch = input_shape_rec_branch
        self.__input_shape_conv_branch = input_shape_conv_branch

        # Build recurrent branch
        self.__recurrent_branch = self._build_recurrent_branch(rec_branch_layers)

        # Build convolutional branch
        self.__conv_branch = self._build_conv_branch(conv_branch_layers, add_repeat_vector_conv_branch)

        # Build tail
        self.__tail = self._build_tail(
            tail_dense_units,
            tail_dense_activation,
            output_dim,
            dropout_dense,
            kernel_regularizer_dense,
            bias_regularizer_dense,
            activity_regularizer_dense,
            kernel_regularizer_softmax,
            bias_regularizer_softmax,
            activity_regularizer_softmax
        )

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

    def _build_tail(self, tail_dense_units: int, tail_dense_activation, output_dim: int, dropout_dense,
                    kernel_regularizer_dense, bias_regularizer_dense, activity_regularizer_dense,
                    kernel_regularizer_softmax, bias_regularizer_softmax, activity_regularizer_softmax) -> Model:
        """
        Builds the tail of the double-branched network.

        :param tail_dense_units: number of units of the tail dense layer.
        :param tail_dense_activation: activation function of the tail dense layer (by default, ReLU is used).
        :param output_dim: number of units/classes of the output layer.
        ::param dropout_dense: dropout rate for the dense layer prior to the last layer.
        :param kernel_regularizer_dense: kernel regularizer for the dense layer prior to the last layer.
        :param bias_regularizer_dense: bias regularizer for the dense layer prior to the last layer.
        :param activity_regularizer_dense: activity regularizer for the dense layer prior to the last layer.
        :param kernel_regularizer_dense: kernel regularizer for the last layer.
        :param bias_regularizer_dense: bias regularizer for the last layer.
        :param activity_regularizer_dense: activity regularizer for the last layer.
        :return: a keras Model representing the tail of the double-branched network.
        """
        merge = TimeDistributed(
            Concatenate(),
            name="tail_concatenate_layer"
        )([self.__recurrent_branch.output, self.__conv_branch.output])
        dense0 = TimeDistributed(
            Dense(
                units=tail_dense_units,
                activation=tail_dense_activation,
                kernel_regularizer=kernel_regularizer_dense,
                bias_regularizer=bias_regularizer_dense,
                activity_regularizer=activity_regularizer_dense,
            ),
            name="tail_dense0"
        )(merge)
        if dropout_dense > 0:
            dense0 = TimeDistributed(
                Dropout(rate=dropout_dense),
                name="tail_dropout_dense"
            )(dense0)
        dense1 = TimeDistributed(
            Dense(
                units=output_dim,
                activation=tail_dense_activation,
                kernel_regularizer=kernel_regularizer_softmax,
                bias_regularizer=bias_regularizer_softmax,
                activity_regularizer=activity_regularizer_softmax
            ),
            name="tail_output_dense"
        )(dense0)
        output = TimeDistributed(
            Softmax(),
            name="tail_output_softmax"
        )(dense1)
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

    def call(self, inputs, training=None, mask=None):
        rec_branch_input, conv_branch_input = inputs
        rec_output = self.__recurrent_branch(rec_branch_input)
        conv_output = self.__conv_branch(conv_branch_input)
        return self.__tail([rec_output, conv_output])

    def get_config(self) -> dict[str, Union[None, list[Optional[dict[str, Any]]], tuple, int]]:
        config_dict = {
            "latent_space_dim": self._latent_space_dim,
            **self.__recurrent_branch.get_config(),
            **self.__conv_branch.get_config(),
            **self.__tail.get_config()
        }
        return config_dict

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
