from typing import Iterable, final, Optional, Union, Sized
import numpy as np
from autoencoder import AutoEncoder
from keras.layers import GRU, LSTM, Layer, BatchNormalization, AveragePooling1D, Flatten, \
    Dense, Dropout, Conv1DTranspose, Reshape

_LAYER_NUM_DEFAULT: final = 3
_LAYER_REDUCTION_FACTOR: final = 0.5
_DEFAULT_INPUT_DIM: final = 256
_UNIT_TYPES: final = {
    "GRU": GRU,
    "LSTM": LSTM
}


class RecurrentAutoEncoder(AutoEncoder):
    """
    This class represents an LSTM or GRU autoencoder model.
    """

    def __init__(self, input_shape: tuple[int, ...], unit_types: list[str], recurrent_units: list[int],
                 latent_space_dim: int, activations: Optional[Union[str, list[str]]] = 'relu',
                 recurrent_activations: Optional[Union[str, list[str]]] = 'sigmoid', bottleneck_unit_type: str = "LSTM",
                 bottleneck_returns_sequences: bool = False, bottleneck_activation: str = 'relu',
                 bottleneck_recurrent_activation: str = 'relu', recurrent_units_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0, recurrent_initializer: str = 'glorot_uniform',
                 kernel_initializer: str = 'orthogonal', bias_initializer: str = 'zeros', recurrent_regularizer=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, go_backwards: bool = False):
        """
        Constructor. Most of the parameters used in keras LSTM/GRU layers can be passed to this method.

        :param input_shape: a integer tuple representing the input shape the model will be build from; this class needs
            at least a 3-dimensional tensor as input shape: (batch_size, timesteps, features).
        :param unit_types: an iterable containing the unit types (either LSTM or GRU).
        :param recurrent_units: an iterable containing the number of units of each recurrent layer.
        :param latent_space_dim: dimensionality of the latent space.
        :param activations: either a single string or a iterable indicating the activations functions to use in each
            layer output. Default: ReLU (relu). If None is given, no activation is applied to the corresponding layer
            (ie. "linear" activation: a(x) = x).
        :param recurrent_activations: either a single string or a iterable indicating the activations functions to use
            in each layer output. Default: ReLU (relu). If None is given, no activation is applied to the corresponding
            layer (ie. "linear" activation: a(x) = x).
        :param bottleneck_unit_type: unit type of the bottleneck (either LSTM or GRU).
        :param bottleneck_activation: a string indicating the activations functions to use in bottleneck layer output.
            Default: ReLU (relu). If None is given, no activation is applied to the corresponding layer (ie. "linear"
            activation: a(x) = x).
        :param recurrent_activations: a string indicating the activations functions to use in bottleneck layer output.
            Default: ReLU (relu). If None is given, no activation is applied to the corresponding layer (ie. "linear"
            activation: a(x) = x).
        :param recurrent_units_dropout: Float between 0 and 1. Fraction of the activation units to drop for the linear
            transformation of the inputs. Default: 0.
        :param recurrent_dropout: Float between 0 and 1. Fraction of the recurrent units to drop for the linear
            transformation of the recurrent state. Default: 0.
        :param recurrent_initializer: Initializer for the recurrent_kernel weights matrix of each layer, used for the
            linear transformation of the recurrent state. Default: orthogonal.
        :param kernel_initializer: Initializer for the kernel weights matrix of each layer, used for the linear
            transformation of the inputs. Default: glorot_uniform.
        :param bias_initializer: Initializer for the bias vector. Default: zeros.
        :param recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix. Default:
            None.
        :param kernel_regularizer: Regularizer function applied to the kernel weights matrix. Default: None.
        :param bias_regularizer: Regularizer function applied to the bias vector. Default: None.
        :param activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
            Default: None.
        :param go_backwards: Boolean (default False). If True, process the input sequence backwards and return the
            reversed sequence.
        """
        if input_shape is not None and input_shape[-1] < 1:
            raise ValueError('Feature number must be strictly positive. '
                             f'Received input_shape={input_shape}.')
        if unit_types is not None and len(unit_types) <= 0:
            raise ValueError('Invalid value for argument `unit_types`. Expected a strictly positive value. '
                             f'Received unit_types={unit_types}.')
        if recurrent_units is not None and len(recurrent_units) <= 0:
            raise ValueError('Invalid value for argument `conv_kernels_size`. Expected a strictly positive value. '
                             f'Received conv_kernels_size={recurrent_units}.')
        if latent_space_dim is not None and latent_space_dim <= 0:
            raise ValueError('Invalid value for argument `latent_space_dim`. Expected a strictly positive value. '
                             f'Received latent_space_dim={latent_space_dim}.')
        if latent_space_dim is not None and latent_space_dim <= 0:
            raise ValueError('Invalid value for argument `latent_space_dim`. Expected a strictly positive value. '
                             f'Received latent_space_dim={latent_space_dim}.')
        if len(recurrent_units) != len(unit_types):
            raise ValueError('Invalid value for argument `unit_types` or `recurrent_units`. Same '
                             'dimension expected.'
                             f'\nReceived conv_filters dimension={len(unit_types)}.'
                             f'\nReceived conv_kernels_size dimension={len(recurrent_units)}.')
        # Setup instance variables
        self._unit_types = unit_types
        self._recurrent_units = recurrent_units

        # If activations or recurrent activations are a single string or None, copy them in an iterable
        if activations is None or isinstance(activations, str):
            self._activations = [activations in range(0, len(recurrent_units))]
        elif isinstance(activations, Iterable):
            self._activations = activations

        if activations is None or isinstance(activations, str):
            self._activations = [recurrent_activations in range(0, len(recurrent_units))]
        elif isinstance(activations, Iterable):
            self._recurrent_activations = recurrent_activations

        self._bottleneck_returns_sequences = bottleneck_returns_sequences
        self._recurrent_units_dropout = recurrent_units_dropout
        self._recurrent_dropout = recurrent_dropout
        self._recurrent_initializer = recurrent_initializer
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._recurrent_regularizer = recurrent_regularizer
        self._go_backwards = go_backwards

        encoder_layers = self._build_encoder_layers()
        bottleneck = self._build_bottleneck(
            latent_space_dim,
            bottleneck_unit_type,
            bottleneck_activation,
            bottleneck_recurrent_activation
        )
        decoder_layers = self._build_decoder_layers()

        super(RecurrentAutoEncoder, self).__init__(
            input_shape=input_shape,
            encoder_layers=encoder_layers,
            bottleneck=bottleneck,
            decoder_layers=decoder_layers,
            outputs_sequences=True
        )

    def _build_encoder_layers(self) -> list[Union[LSTM, GRU]]:
        """
        Builds a new recurrent  block, composed of a LSTM or a GRU layers

        :return: created LSTM or GRU block
        """
        encoder_layers: list[Union[LSTM, GRU]] = []

        for layer_index in range(len(self._unit_types)):

            unit_type = self._unit_types[layer_index]
            constructor = _UNIT_TYPES[unit_type]
            units = self._recurrent_units[layer_index]
            activation = self._activations[layer_index]
            recurrent_activation = self._recurrent_activations[layer_index]

            # Construct either LSTM or GRU
            recurrent_layer = constructor(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                recurrent_initializer=self._recurrent_initializer,
                recurrent_regularizer=self._recurrent_regularizer,
                dropout=self._recurrent_units_dropout,
                recurrent_dropout=self._recurrent_dropout,
                return_sequences=True,
                go_backwards=self._go_backwards
            )
            encoder_layers.append(recurrent_layer)

        return encoder_layers

    def _build_bottleneck(self, latent_space_dim: int, unit_type: str, activation: str, recurrent_activation: str) \
            -> Union[LSTM, GRU]:
        """
        Build the bottleneck layer that consist of a LSTM or a GRU layer

        :param latent_space_dim: An integer. Dimensionality of the bottleneck.
        :param unit_type: A string. Either "LSTM" or "GRU" to chose the type of layer
        :param activation: Activation function to use. If you don't specify anything, no activation is applied.
        :param recurrent_activation: Activation function to use for the recurrent step.
        :return: created LSTM or GRU block
        """
        constructor = _UNIT_TYPES[unit_type]
        units = latent_space_dim

        # Construct either LSTM or GRU
        bottleneck = constructor(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            recurrent_initializer=self._recurrent_initializer,
            recurrent_regularizer=self._recurrent_regularizer,
            dropout=self._recurrent_units_dropout,
            recurrent_dropout=self._recurrent_dropout,
            return_sequences=True,
            go_backwards=self._go_backwards
        )

        return bottleneck

    def _build_decoder_layers(self) -> list[Union[LSTM, GRU]]:
        """
        Build the decoder layer, usually symmetrical to the decoder

        :return: created decoder block
        """
        decoder_layers: list[Union[LSTM, GRU]] = []

        for layer_index in reversed(range(len(self._unit_types))):

            unit_type = self._unit_types[layer_index]
            constructor = _UNIT_TYPES[unit_type]
            units = self._recurrent_units[layer_index]
            activation = self._activations[layer_index]
            recurrent_activation = self._recurrent_activations[layer_index]

            # Construct either LSTM or GRU
            recurrent_layer = constructor(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_initializer=self._bias_initializer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                recurrent_initializer=self._recurrent_initializer,
                recurrent_regularizer=self._recurrent_regularizer,
                dropout=self._recurrent_units_dropout,
                recurrent_dropout=self._recurrent_dropout,
                return_sequences=self._bottleneck_returns_sequences,
                go_backwards=self._go_backwards
            )
            decoder_layers.append(recurrent_layer)

        return decoder_layers

