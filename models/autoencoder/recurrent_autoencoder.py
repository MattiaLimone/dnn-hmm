import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, RepeatVector, TimeDistributed
import keras
from typing import final, Optional

_LAYER_NUM_DEFAULT: final = 3
_LAYER_REDUCTION_FACTOR: final = 0.5
_DEFAULT_INPUT_DIM: final = 256
_MODEL_NAME: final = "RUMBLING_AUTOENCODER"
_UNIT_TYPES: final = {
    "GRU": GRU,
    "LSTM": LSTM
}


# TODO: this class must inherit from keras.models.Sequential
class RecurrentAutoEncoder(Sequential):
    def __init__(self, n_encoding_layers: int = _LAYER_NUM_DEFAULT, input_neurons: int = _DEFAULT_INPUT_DIM,
                 encoding_dims: Optional[list] = None, name: str = _MODEL_NAME, unit_type: str = "LSTM",
                 activation='relu', dropout: float = 0, recurrent_dropout: float = 0, recurrent_regularizer=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, recurrent_constraint=None,
                 kernel_constraint=None, bias_constraint=None, ):
        super().__init__(name)
        if n_encoding_layers <= 0:
            raise ValueError("n_encoding_layers must be positive")
        if input_neurons <= 0:
            raise ValueError("input_neurons number must be positive")
        if encoding_dims is not None and encoding_dims != len(encoding_dims):
            raise ValueError("encoding_dims length must be equal to n_encoding_layers")
        if encoding_dims is not None and encoding_dims[0] != input_neurons:
            raise ValueError("encoding_dims first element must be equal to input neurons")
        if unit_type not in _UNIT_TYPES:
            raise ValueError("unit_type must be either: " + str(_UNIT_TYPES.values()))
        if not 0 <= recurrent_dropout < 1:
            raise ValueError("recurrent_dropout must be between 0 and 1 (excluded)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1 (excluded)")


        # Reference to the layer class (not instance)
        self._model_class_constructor = _UNIT_TYPES[unit_type]

        # If dims are None generate them
        if encoding_dims is None:
            encoding_dims = []
            n_units = input_neurons
            for i in range(0, n_encoding_layers):
                encoding_dims.append(n_units)
                n_units /= 2

        # Add layers
        for n_units in encoding_dims:
            LSTM(units=n_units, activation=activation, return_sequences=True)
            self.add(self._model_class_constructor())

