from typing import Optional, Any, Union, final
from keras.layers import RNN, BatchNormalization, Normalization, LayerNormalization, Conv1D, TimeDistributed, \
    Concatenate, Dense, Softmax, RepeatVector, Flatten, Layer
from keras.layers.pooling import Pooling1D
from keras.models import Model, Sequential


_RECURRENT_BRANCH_NAME: final = "recurrent_branch"
_CONV_BRANCH_NAME: final = "conv_branch"
_TAIL_NAME: final = "tail"


# TODO: write class and methods documentation
class RecConv1DSiameseNet(Model):

    def __init__(self,
                 rec_branch_layers: list[RNN, BatchNormalization, RepeatVector, LayerNormalization, Layer],
                 conv_branch_layers: list[Conv1D, BatchNormalization, LayerNormalization, Pooling1D, Layer],
                 input_shape_rec_branch: tuple[Optional[int], int, ...],
                 input_shape_conv_branch: tuple[Optional[int], int, ...], tail_dense_units: int, output_dim: int,
                 tail_dense_activation='relu', timesteps_repeat_vector_conv_branch: Optional[int] = None):

        # TODO: add checks to the input shape of each branch (for example, rec branch inputs must be at least 3D
        super().__init__()

        # Set instance variables
        self.__input_shape_rec_branch = input_shape_rec_branch
        self.__input_shape_conv_branch = input_shape_conv_branch

        # Build recurrent branch
        self.__recurrent_branch = self._build_recurrent_branch(rec_branch_layers)

        # Build convolutional branch
        self.__conv_branch = self._build_conv_branch(conv_branch_layers, timesteps_repeat_vector_conv_branch)

        # Build tail
        self.__tail = self._build_tail(tail_dense_units, tail_dense_activation, output_dim)

        # Build the model
        self.build(input_shape=[input_shape_rec_branch, input_shape_conv_branch])

    def _build_recurrent_branch(
            self,
            rec_branch_layers: list[RNN, BatchNormalization, Normalization, LayerNormalization]
    ) -> Sequential:
        recurrent_branch = Sequential(name=_RECURRENT_BRANCH_NAME)

        # Add all the given layers to the recurrent branch Sequential model
        for rec_branch_layer in rec_branch_layers:
            recurrent_branch.add(rec_branch_layer)

        # Build the recurrent branch
        recurrent_branch.build(input_shape=self.__input_shape_rec_branch)
        return recurrent_branch

    def _build_conv_branch(
            self,
            conv_branch_layers: list[Conv1D, BatchNormalization, Normalization, LayerNormalization, Pooling1D],
            timesteps_repeat_vector_conv_branch: Optional[int]
    ) -> Sequential:
        conv_branch = Sequential(name=_CONV_BRANCH_NAME)

        # Add all the given layers to the convolutional branch Sequential model
        for conv_branch_layer in conv_branch_layers:
            conv_branch.add(conv_branch_layer)

        # If required, add flatten and repeat vector layers to allow concatenation with recurrent branch frames overtime
        if timesteps_repeat_vector_conv_branch is not None:
            conv_branch.add(Flatten(name="conv_branch_flatten"))
            conv_branch.add(RepeatVector(n=timesteps_repeat_vector_conv_branch, name="conv_branch_repeat_vector"))

        # Build the convolutional branch
        conv_branch.build(input_shape=self.__input_shape_conv_branch)
        return conv_branch

    def _build_tail(self, tail_dense_units: int, tail_dense_activation, output_dim: int) -> Model:
        merge = TimeDistributed(
            Concatenate(),
            name="tail_concatenate_layer"
        )([self.__recurrent_branch.output, self.__conv_branch.output])
        dense0 = TimeDistributed(
            Dense(units=tail_dense_units, activation=tail_dense_activation),
            name="tail_dense0"
        )(merge)
        dense1 = TimeDistributed(
            Dense(units=output_dim, activation=tail_dense_activation),
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
