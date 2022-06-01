from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, RepeatVector, TimeDistributed
from typing import final, Optional


_LAYER_NUM_DEFAULT: final = 3
_LAYER_REDUCTION_FACTOR: final = 0.5
_DEFAULT_INPUT_DIM: final = 256
_UNIT_TYPES: final = {
    "GRU": GRU,
    "LSTM": LSTM
}


class RecurrentAutoEncoder(Sequential):
    """
    Generates a compiled LSTM or GRU autoencoder model. Most of the parameters used in keras LSTM/GRU can be passed to
    this class.
    """
    def __init__(self, n_encoding_layers: int = _LAYER_NUM_DEFAULT, input_neurons: int = _DEFAULT_INPUT_DIM,
                 timesteps=None, n_features=None, encoding_dims: Optional[list] = None,
                 sequential_bottleneck: bool = False, unit_type: str = "LSTM", activation='relu',
                 recurrent_activation: str = 'sigmoid', dropout: float = 0, recurrent_dropout: float = 0,
                 recurrent_initializer: str = 'glorot_uniform', kernel_initializer: str = 'orthogonal',
                 bias_initializer: str = 'zeros', recurrent_regularizer=None, kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, recurrent_constraint=None, kernel_constraint=None,
                 bias_constraint=None, return_sequences: bool = True, return_state: bool = False,
                 go_backwards: bool = False, stateful: bool = False, time_major: bool = False):
        """
        Constructor

        :param n_encoding_layers: Positive integer, dimensionality of the encoder/decoder
        :param input_neurons: Positive integer, dimensionality of the output space of the first layer
        :param timesteps: Number of elements of each sequence. Use in the Repeat Vector layer
        :param n_features: Number of features of each timestep. Used for the Time Distributed layer
        :param encoding_dims: List of dimension of each layer of the encoder/decoder. Size must be the same 
            as n_encoding_layers
        :param unit_type: Decider of the layer type, either LSTM or GRU
        :param activation: Activation function to use. Default: hyperbolic tangent (tanh). If you pass None, no 
            activation is applied (ie. "linear" activation: a(x) = x)
        :param recurrent_activation: Activation function to use for the recurrent step. Default: sigmoid (sigmoid). 
            If you pass None, no activation is applied (ie. "linear" activation: a(x) = x)
        :param dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the 
            inputs. Default: 0
        :param recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of 
            the recurrent state. Default: 0
        :param recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear 
            transformation of the recurrent state. Default: orthogonal
        :param kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of 
            the inputs. Default: glorot_uniform
        :param bias_initializer: Initializer for the bias vector. Default: zeros
        :param recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix. Default: None
        :param kernel_regularizer: Regularizer function applied to the kernel weights matrix. Default: None
        :param bias_regularizer: Regularizer function applied to the bias vector. Default: None
        :param activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). 
            Default: None
        :param recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix. Default: None
        :param kernel_constraint: Constraint function applied to the kernel weights matrix. Default: None
        :param bias_constraint: Constraint function applied to the bias vector. Default: None
        :param return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full 
            sequence. Default: False.
        :param return_state: Boolean. Whether to return the last state in addition to the output. Default: False
        :param go_backwards: Boolean (default False). If True, process the input sequence backwards and return the 
            reversed sequence
        :param stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will 
            be used as initial state for the sample of index i in the following batch
        :param time_major: The shape format of the inputs and outputs tensors. If True, the inputs and outputs will be 
            in shape [timesteps, batch, feature], whereas in the False case, it will be [batch, timesteps, feature]. 
            Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input 
            and emits output in batch-major form
        """
        super().__init__()
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
        if timesteps is None:
            raise ValueError("timesteps must be specified")
        if n_features is None:
            raise ValueError("n_features must be specified")

        # Reference to the layer class (not instance)
        self._model_class_constructor = _UNIT_TYPES[unit_type]
        # Input shape of the first layer
        input_shape = (timesteps, n_features)

        # If dims are None generate them
        if encoding_dims is None:
            encoding_dims = []
            n_units = input_neurons
            for i in range(0, n_encoding_layers):
                encoding_dims.append(n_units)
                n_units = int(n_units / 2)

        # Adding the other layers specified in encoding_dims list. Last layer has return_sequences parameters set to
        # false in order to have a vector of features to pass to the Repeat Vector layer
        for i, n_units in enumerate(encoding_dims):
            # If sequential_bottleneck is true then last layer must give a feature vector as output so it doesn't have 
            # to return the entire sequence
            if i == n_encoding_layers - 1 and sequential_bottleneck:
                return_sequences = False
            # Adding first layer, input_shape must be specified in this layer
            if i == 0:
                self.add(self._model_class_constructor(
                    units=input_neurons,
                    activation=activation,
                    input_shape=input_shape,
                    return_sequences=return_sequences,
                    recurrent_activation=recurrent_activation,
                    use_bias=True,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    recurrent_constraint=recurrent_constraint,
                    bias_constraint=bias_constraint,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_state=return_state,
                    go_backwards=go_backwards,
                    stateful=stateful,
                    time_major=time_major
                    )
                )
            # otherwise it's an inner encoder layer without input_shape parameter
            else:
                self.add(self._model_class_constructor(
                    units=n_units,
                    activation=activation,
                    return_sequences=return_sequences,
                    recurrent_activation=recurrent_activation,
                    use_bias=True,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    recurrent_constraint=recurrent_constraint,
                    bias_constraint=bias_constraint,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_state=return_state,
                    go_backwards=go_backwards,
                    stateful=stateful,
                    time_major=time_major
                    )
                )
        # Reversing layer dimensionality list to build the decoder specular to the encoder
        encoding_dims.reverse()
        # Adding the Repeat Vector layer. It transforms a (None, n_feature) shape into a (None, timesteps, n_features)
        if sequential_bottleneck:
            self.add(RepeatVector(timesteps))
        # Adding the decoders layer in reversed order of the encoder
        for n_units in encoding_dims:
            self.add(self._model_class_constructor(
                units=n_units,
                activation=activation,
                return_sequences=True,
                recurrent_activation=recurrent_activation,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_state=return_state,
                go_backwards=go_backwards,
                stateful=stateful,
                time_major=time_major
                )
            )
        # Adding last layer that has the same size as the input of the network
        self.add(TimeDistributed(Dense(n_features)))
