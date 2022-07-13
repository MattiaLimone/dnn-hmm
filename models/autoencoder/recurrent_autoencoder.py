from typing import Iterable, final, Optional, Union, Any
from models.autoencoder.autoencoder import AutoEncoder
from keras.layers import GRU, LSTM, RepeatVector, Layer
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='recurrent_autoencoder')
class LSTMRepeatVector(Layer):
    """
    This class represents a simple neural network layer composed of an LSTM layer and a RepeatVector layer.
    """
    def __init__(self, units: int, repeat_vector_timesteps: int, activation='tanh', recurrent_activation='sigmoid',
                 use_bias: bool = True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias: bool = True, kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None, dropout: float = 0., recurrent_dropout: float = 0.,
                 return_state: bool = False, go_backwards: bool = False, stateful: bool = False, unroll: bool = False,
                 name: Optional[str] = None, rec_name: Optional[str] = None, repeat_vector_name: Optional[str] = None,
                 **kwargs):
        """Long Short-Term Memory layer - Hochreiter 1997, with a RepeatVector layer stacked on top.

        Note that this cell is not optimized for performance on GPU. Please use
        `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.

        Args:
        units: Positive integer, dimensionality of the output space.
        repeat_vector_timesteps: Positive integer, represents the number of timesteps in which the output vector of the
            LSTM cell will be repeated.
        activation: Activation function to use.
          Default: hyperbolic tangent (`tanh`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
          for the recurrent step.
          Default: hard sigmoid (`hard_sigmoid`).
          If you pass `None`, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs..
        recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean.
          If True, add 1 to the bias of the forget gate at initialization.
          Setting it to true will also force `bias_initializer="zeros"`.
          This is recommended in [Jozefowicz et al., 2015](
            https://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
        return_state: Boolean. Whether to return the last state
          in addition to the output.
        go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
        stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
        unroll: Boolean (default False).
          If True, the network will be unrolled,
          else a symbolic loop will be used.
          Unrolling can speed-up a RNN,
          although it tends to be more memory-intensive.
          Unrolling is only suitable for short sequences.
        time_major: The shape format of the `inputs` and `outputs` tensors.
          If True, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
        name: a string (default None) representing the name of the layer.
        rec_name: a string (default None) representing the name of the LSTM cell composing the layer.
        repeat_vector_name: a string (default None) representing the name of the RepeatVector part of the layer.

        Call arguments:
        inputs: a 3D tensor.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
          a given timestep should be masked. An individual `True` entry indicates
          that the corresponding timestep should be utilized, while a `False`
          entry indicates that the corresponding timestep should be ignored.
        training: Python boolean indicating whether the layer should behave in
          training mode or in inference mode. This argument is passed to the cell
          when calling it. This is only relevant if `dropout` or
          `recurrent_dropout` is used.
        initial_state: List of initial state tensors to be passed to the first
          call of the cell.
        """
        super(LSTMRepeatVector, self).__init__(trainable=True, name=name)
        self.__lstm = LSTM(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
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
            unroll=unroll,
            name=rec_name,
            **kwargs
        )
        self.__repeat_vector = RepeatVector(name=repeat_vector_name, n=repeat_vector_timesteps)

    def call(self, inputs, *args, **kwargs):
        """
        Calls the model on new inputs and returns the outputs as tensors, feeding the LSTM cell and then repeating the
        output vector n times.

        :param inputs: Input tensor, or dict/list/tuple of input tensors.
        :param args: Additional positional arguments. May contain tensors, although this is not recommended.
        :param kwargs: Additional keyword arguments. May contain tensors, although this is not recommended.
        :return: a tensor or list/tuple of tensors containing the output of the LSTM cell repeated n times.
        """
        lstm_output = self.__lstm(inputs)
        repeat_vector_output = self.__repeat_vector(lstm_output)
        return repeat_vector_output

    def build(self, input_shape):
        """
        Creates the variables of the layer.

        :param input_shape: Instance of `TensorShape`, or list of instances of `TensorShape` if the layer expects a list
            of inputs (one instance per input).
        """
        self.__lstm.build(input_shape)
        lstm_output_shape = self.__lstm.compute_output_shape(input_shape)
        self.__repeat_vector.build(lstm_output_shape)
        super(LSTMRepeatVector, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        :param input_shape: hape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions, instead of an integer.
        :return: An input shape tuple.
        """
        lstm_output_shape = self.__lstm.compute_output_shape(input_shape)
        return self.__repeat_vector.compute_output_shape(lstm_output_shape)

    def get_config(self):
        config = super(LSTMRepeatVector, self).get_config()
        repeat_vector_timesteps = self.__repeat_vector.n
        repeat_vector_name = self.__repeat_vector.name
        rec_name = self.__lstm.name
        rec_config = self.__lstm.get_config()
        del rec_config['name']
        config.update({
            "repeat_vector_timesteps": repeat_vector_timesteps,
            "repeat_vector_name": repeat_vector_name,
            "rec_name": rec_name,
            **rec_config
        })
        return config

    @property
    def units(self) -> int:
        """
        Returns the units number of the LSTM layer.

        :return: An integer representing the number of units of the LSTM layer.
        """
        return self.__lstm.units


@tf.keras.utils.register_keras_serializable(package='recurrent_autoencoder')
class GRURepeatVector(Layer):
    """
    This class represents a simple neural network layer composed of an LSTM layer and a RepeatVector layer.
    """
    def __init__(self, units: int, repeat_vector_timesteps: int, activation='tanh', recurrent_activation='sigmoid',
                 use_bias: bool = True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                 dropout: float = 0., recurrent_dropout: float = 0., return_state: bool = False,
                 go_backwards: bool = False, stateful: bool = False, unroll: bool = False, reset_after: bool = True,
                 name: Optional[str] = None, rec_name: Optional[str] = None, repeat_vector_name: Optional[str] = None,
                 **kwargs):
        """Gated Recurrent Unit - Cho et al. 2014,  with a RepeatVector layer stacked on top.

        There are two variants. The default one is based on 1406.1078v3 and
        has reset gate applied to hidden state before matrix multiplication. The
        other one is based on original 1406.1078v1 and has the order reversed.

        The second variant is compatible with CuDNNGRU (GPU-only) and allows
        inference on CPU. Thus it has separate biases for `kernel` and
        `recurrent_kernel`. Use `'reset_after'=True` and
        `recurrent_activation='sigmoid'`.

        Args:
          units: Positive integer, dimensionality of the output space.
          repeat_vector_timesteps: Positive integer, represents the number of timesteps in which the output vector of
          the GRU cell will be repeated.
          activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
          recurrent_activation: Activation function to use
            for the recurrent step.
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
          recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent state.
          bias_initializer: Initializer for the bias vector.
          kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
          recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
          kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
          recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.
          dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
          recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
          return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
          return_state: Boolean. Whether to return the last state
            in addition to the output.
          go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
          stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
          unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
          time_major: The shape format of the `inputs` and `outputs` tensors.
            If True, the inputs and outputs will be in shape
            `(timesteps, batch, ...)`, whereas in the False case, it will be
            `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
            efficient because it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major, so by
            default this function accepts input and emits output in batch-major
            form.
          reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (cuDNN compatible).
          name: a string (default None) representing the name of the layer.
          rec_name: a string (default None) representing the name of the GRU cell composing the layer.
          repeat_vector_name: a string (default None) representing the name of the RepeatVector part of the layer.

        Call arguments:
          inputs: a 3D tensor.
          mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked. An individual `True` entry indicates
            that the corresponding timestep should be utilized, while a `False`
            entry indicates that the corresponding timestep should be ignored.
          training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the cell
            when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used.
          initial_state: List of initial state tensors to be passed to the first
            call of the cell.
        """

        super(GRURepeatVector, self).__init__(trainable=True, name=name)
        self.__gru = GRU(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
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
            unroll=unroll,
            reset_after=reset_after,
            name=rec_name,
            **kwargs
        )
        self.__repeat_vector = RepeatVector(name=repeat_vector_name, n=repeat_vector_timesteps)

    def call(self, inputs, *args, **kwargs):
        """
        Calls the model on new inputs and returns the outputs as tensors, feeding the GRU cell and then repeating the
        output vector n times.

        :param inputs: Input tensor, or dict/list/tuple of input tensors.
        :param args: Additional positional arguments. May contain tensors, although this is not recommended.
        :param kwargs: Additional keyword arguments. May contain tensors, although this is not recommended.
        :return: a tensor or list/tuple of tensors containing the output of the GRU cell repeated n times.
        """
        gru_output = self.__gru(inputs)
        repeat_vector_output = self.__repeat_vector(gru_output)
        return repeat_vector_output

    def build(self, input_shape):
        """
        Creates the variables of the layer.

        :param input_shape: Instance of `TensorShape`, or list of instances of `TensorShape` if the layer expects a list
            of inputs (one instance per input).
        """
        self.__gru.build(input_shape)
        gru_output_shape = self.__gru.compute_output_shape(input_shape)
        self.__repeat_vector.build(gru_output_shape)
        super(GRURepeatVector, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        :param input_shape: hape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions, instead of an integer.
        :return: An input shape tuple.
        """
        gru_output_shape = self.__gru.compute_output_shape(input_shape)
        return self.__repeat_vector.compute_output_shape(gru_output_shape)

    def get_config(self):
        config = super(GRURepeatVector, self).get_config()
        repeat_vector_timesteps = self.__repeat_vector.n
        repeat_vector_name = self.__repeat_vector.name
        rec_name = self.__gru.name
        rec_config = self.__gru.get_config()
        del rec_config['name']
        config.update({
            "repeat_vector_timesteps": repeat_vector_timesteps,
            "repeat_vector_name": repeat_vector_name,
            "rec_name": rec_name,
            **rec_config
        })
        return config

    @property
    def units(self) -> int:
        """
        Returns the units number of the GRU layer.

        :return: An integer representing the number of units of the GRU layer.
        """
        return self.__gru.units


_UNIT_TYPES: final = {
    "GRU": GRU,
    "LSTM": LSTM
}

_UNIT_TYPES_RV: final = {
    "GRU": GRURepeatVector,
    "LSTM": LSTMRepeatVector
}


class RecurrentAutoEncoder(AutoEncoder):
    """
    This class represents an LSTM or GRU autoencoder model.
    """

    def __init__(self, input_shape: tuple[int, ...], unit_types: list[str], recurrent_units: list[int],
                 latent_space_dim: int, activations: Optional[Union[str, list[str]]] = 'tanh',
                 recurrent_activations: Optional[Union[str, list[str]]] = 'sigmoid',
                 bottleneck_unit_type: str = "LSTM", bottleneck_returns_sequences: bool = False,
                 bottleneck_activation: str = 'tanh', bottleneck_recurrent_activation: str = 'sigmoid',
                 bottleneck_recurrent_regularizer=None, bottleneck_kernel_regularizer=None,
                 bottleneck_bias_regularizer=None, bottleneck_activity_regularizer=None,
                 recurrent_units_dropout: float = 0.0, recurrent_dropout: float = 0.0,
                 recurrent_initializer: str = 'glorot_uniform', kernel_initializer: str = 'orthogonal',
                 bias_initializer: str = 'zeros', recurrent_regularizer: Optional[list[None, Any]] = None,
                 kernel_regularizer: Optional[list[None, Any]] = None,
                 bias_regularizer: Optional[list[None, Any]] = None,
                 activity_regularizer: Optional[list[None, Any]] = None, go_backwards: bool = False,
                 do_batch_norm: bool = True, last_layer_activation=None, last_layer_kernel_regularizer=None,
                 last_layer_bias_regularizer=None, last_layer_activity_regularizer=None):
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
        :param bottleneck_recurrent_regularizer: recurrent kernel regularizer for the bottleneck layer. Useful to make
            the autoencoder sparse (by default, no regularizer is used).
        :param bottleneck_kernel_regularizer: kernel regularizer for the bottleneck layer. Useful to make
            the autoencoder sparse (by default, no regularizer is used).
        :param bottleneck_bias_regularizer: bias regularizer for the bottleneck layer. Useful to make
            the autoencoder sparse (by default, no regularizer is used).
        :param bottleneck_activity_regularizer: activity regularizer for the bottleneck layer. Useful to make the
            autoencoder sparse (by default, no regularizer is used).
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
        :param recurrent_regularizer: List of regularizer functions applied to the recurrent_kernel weights matrix of
            the decoder layers, from the last to the first (a list of None and regularizer functions can be given
            too). Default: None.
        :param kernel_regularizer: List of regularizer functions applied to the kernel weights matrix of the decoder
            layers, from the last to the first (a list of None and regularizer functions can be given too). Default:
            None.
        :param bias_regularizer: List of regularizer functions applied to the bias vector of the decoder layers, from
            the last to the first (a list of None and regularizer functions can be given too). Default: None.
        :param activity_regularizer: List of regularizer functions applied to the output of the decoder layers (their
            "activation"), , from the last to the first; a list of None and regularizer functions can be given too.
            Default: None.
        :param go_backwards: Boolean (default False). If True, process the input sequence backwards and return the
            reversed sequence.
        :param do_batch_norm: whether or not to add a batch normalization layer before the output layer of the decoder.
        :param last_layer_activation: activation function applied to the output layer, if None is given, then a linear
                                      activation function is applied (a(x) = x).
        :param last_layer_kernel_regularizer: regularization function applied to the output layer's kernel.
        :param last_layer_bias_regularizer: regularization function applied to the output layer's biases.
        :param last_layer_activity_regularizer: regularization function applied to the output layer's activation.
        """
        if not input_shape and input_shape[-1] < 1:
            raise ValueError('Feature number must be strictly positive. '
                             f'Received input_shape={input_shape}.')
        if not unit_types and len(unit_types) <= 0:
            raise ValueError('Invalid value for argument `unit_types`. Expected a strictly positive value. '
                             f'Received unit_types={unit_types}.')
        if not recurrent_units and len(recurrent_units) <= 0:
            raise ValueError('Invalid value for argument `recurrent_units`. Expected a strictly positive value. '
                             f'Received recurrent_units={recurrent_units}.')
        if not latent_space_dim and latent_space_dim <= 0:
            raise ValueError('Invalid value for argument `latent_space_dim`. Expected a strictly positive value. '
                             f'Received latent_space_dim={latent_space_dim}.')
        if len(recurrent_units) != len(unit_types):
            raise ValueError('Invalid value for argument `unit_types` or `recurrent_units`. Same dimension expected.'
                             f'\nReceived len(unit_types)={len(unit_types)}.'
                             f'\nReceived len(recurrent_units)={len(recurrent_units)}.')
        if recurrent_regularizer is not None and len(recurrent_regularizer) != len(unit_types):
            raise ValueError('Invalid value for argument `recurrent_regularizer`. Same dimension as `unit_types` '
                             'expected if given not None.' f'\nReceived len(unit_types)={len(unit_types)}.'
                             f'\nReceived len(recurrent_regularizer)={len(recurrent_regularizer)}.')
        if kernel_regularizer is not None and len(kernel_regularizer) != len(unit_types):
            raise ValueError('Invalid value for argument `kernel_regularizer`. Same dimension as `unit_types` '
                             'expected if given not None.' f'\nReceived len(unit_types)={len(unit_types)}.'
                             f'\nReceived len(kernel_regularizer)={len(kernel_regularizer)}.')
        if bias_regularizer is not None and len(bias_regularizer) != len(unit_types):
            raise ValueError('Invalid value for argument `bias_regularizer`. Same dimension as `unit_types` '
                             'expected if given not None.' f'\nReceived len(unit_types)={len(unit_types)}.'
                             f'\nReceived len(bias_regularizer)={len(bias_regularizer)}.')
        if activity_regularizer is not None and len(activity_regularizer) != len(unit_types):
            raise ValueError('Invalid value for argument `activity_regularizer`. Same dimension as `unit_types` '
                             'expected if given not None.' f'\nReceived len(unit_types)={len(unit_types)}.'
                             f'\nReceived len(activity_regularizer)={len(activity_regularizer)}.')

        # Setup instance variables
        self._unit_types = unit_types
        self._recurrent_units = recurrent_units

        # If activations or recurrent activations are a single string or None, copy them in an iterable
        if activations is None or isinstance(activations, str):
            self._activations = [activations for _ in range(0, len(recurrent_units))]
        elif isinstance(activations, Iterable):
            self._activations = activations

        if activations is None or isinstance(recurrent_activations, str):
            self._recurrent_activations = [recurrent_activations for _ in range(0, len(recurrent_units))]
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
        self._bottleneck_unit_type = bottleneck_unit_type
        self._bottleneck_activation = bottleneck_activation
        self._bottleneck_recurrent_activation = bottleneck_recurrent_activation
        self._bottleneck_recurrent_regularizer = bottleneck_recurrent_regularizer
        self._bottleneck_kernel_regularizer = bottleneck_kernel_regularizer
        self._bottleneck_bias_regularizer = bottleneck_bias_regularizer
        self._bottleneck_activity_regularizer = bottleneck_activity_regularizer

        # Build encoder, bottleneck and decoder layers
        encoder_layers = self._build_encoder_layers()
        timesteps = input_shape[1]
        bottleneck = self._build_bottleneck(
            latent_space_dim,
            bottleneck_unit_type,
            bottleneck_activation,
            bottleneck_recurrent_activation,
            bottleneck_activity_regularizer,
            timesteps
        )
        decoder_layers = self._build_decoder_layers(
            latent_space_dim,
            bottleneck_unit_type,
            bottleneck_activation,
            bottleneck_recurrent_activation
        )

        super(RecurrentAutoEncoder, self).__init__(
            input_shape=input_shape,
            encoder_layers=encoder_layers,
            bottleneck=bottleneck,
            decoder_layers=decoder_layers,
            do_batch_norm=do_batch_norm,
            last_layer_activation=last_layer_activation,
            last_layer_kernel_regularizer=last_layer_kernel_regularizer,
            last_layer_bias_regularizer=last_layer_bias_regularizer,
            last_layer_activity_regularizer=last_layer_activity_regularizer,
            outputs_sequences=True
        )

    def _build_encoder_layers(self) -> list[Union[LSTM, GRU]]:
        """
        Build the encoder block

        :return: list of created LSTM or GRU encoder blocks
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
                name=f"decoder_{unit_type.lower()}{layer_index + 1}",
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=None,
                bias_initializer=self._bias_initializer,
                bias_regularizer=None,
                activity_regularizer=None,
                recurrent_initializer=self._recurrent_initializer,
                recurrent_regularizer=None,
                dropout=self._recurrent_units_dropout,
                recurrent_dropout=self._recurrent_dropout,
                return_sequences=True,
                go_backwards=self._go_backwards
            )
            encoder_layers.append(recurrent_layer)

        return encoder_layers

    def _build_bottleneck(self, latent_space_dim: int, unit_type: str, activation: str, recurrent_activation: str,
                          bottleneck_activity_regularizer, timesteps: int) -> Union[LSTM, GRU]:
        """
        Build the bottleneck layer that consist of a LSTM or a GRU layer

        :param latent_space_dim: An integer. Dimensionality of the bottleneck.
        :param unit_type: a string. Either "LSTM" or "GRU" to chose the type of layer
        :param activation: Activation function to use. If you don't specify anything, no activation is applied.
        :param recurrent_activation: Activation function to use for the recurrent step.
        :param bottleneck_activity_regularizer: activity regularizer for the bottleneck layer. Useful to make the
            autoencoder sparse.
        :param timesteps: number of timesteps.
        :return: created LSTM or GRU bottleneck layer.
        """
        constructor = _UNIT_TYPES[unit_type]
        units = latent_space_dim

        # Construct either LSTM or GRU
        if self._bottleneck_returns_sequences:
            bottleneck = constructor(
                name=f"bottleneck_{unit_type.lower()}",
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._bottleneck_kernel_regularizer,
                bias_initializer=self._bias_initializer,
                bias_regularizer=self._bottleneck_bias_regularizer,
                activity_regularizer=bottleneck_activity_regularizer,
                recurrent_initializer=self._recurrent_initializer,
                recurrent_regularizer=self._bottleneck_recurrent_regularizer,
                dropout=self._recurrent_units_dropout,
                recurrent_dropout=self._recurrent_dropout,
                return_sequences=True,
                go_backwards=self._go_backwards
            )
        else:
            # If bottleneck doesn't return sequences, add a RepeatVector layer prior to the decoder
            constructor = _UNIT_TYPES_RV[unit_type]
            bottleneck = constructor(
                name="bottleneck_layer",
                rec_name=f"bottleneck_{unit_type.lower()}",
                repeat_vector_name="bottleneck_repeat_vector",
                units=units,
                repeat_vector_timesteps=timesteps,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._bottleneck_kernel_regularizer,
                bias_initializer=self._bias_initializer,
                bias_regularizer=self._bottleneck_bias_regularizer,
                activity_regularizer=bottleneck_activity_regularizer,
                recurrent_initializer=self._recurrent_initializer,
                recurrent_regularizer=self._bottleneck_recurrent_regularizer,
                dropout=self._recurrent_units_dropout,
                recurrent_dropout=self._recurrent_dropout,
                go_backwards=self._go_backwards
            )

        return bottleneck

    def _build_decoder_layers(self, latent_space_dim: int, bottleneck_unit_type: str, bottleneck_activation: str,
                              bottleneck_recurrent_activation: str) -> list[Union[LSTM, GRU]]:
        """
        Build the recurrent decoder block.

        :param latent_space_dim: An integer. Dimensionality of the bottleneck.
        :param bottleneck_unit_type: a string. Either "LSTM" or "GRU" to chose the type of layer of the bottleneck.
        :param bottleneck_activation: Activation function used in bottleneck. If you don't specify anything, no
            activation is applied.
        :param bottleneck_recurrent_activation: Activation function to used in the bottleneck for the recurrent step.

        :return: list of created LSTM or GRU decoder blocks.
        """
        decoder_layers: list[Union[LSTM, GRU]] = []
        constructor = _UNIT_TYPES[bottleneck_unit_type]

        bottleneck_mirror = constructor(
            name=f"decoder_{bottleneck_unit_type.lower()}{1}",
            units=latent_space_dim,
            activation=bottleneck_activation,
            recurrent_activation=bottleneck_recurrent_activation,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._bottleneck_kernel_regularizer,
            bias_initializer=self._bias_initializer,
            bias_regularizer=self._bottleneck_bias_regularizer,
            activity_regularizer=self._bottleneck_activity_regularizer,
            recurrent_initializer=self._recurrent_initializer,
            recurrent_regularizer=self._bottleneck_recurrent_regularizer,
            dropout=self._recurrent_units_dropout,
            recurrent_dropout=self._recurrent_dropout,
            return_sequences=True,
            go_backwards=self._go_backwards
        )
        decoder_layers.append(bottleneck_mirror)

        for layer_index in reversed(range(len(self._unit_types))):

            unit_type = self._unit_types[layer_index]
            constructor = _UNIT_TYPES[unit_type]
            units = self._recurrent_units[layer_index]
            activation = self._activations[layer_index]
            recurrent_activation = self._recurrent_activations[layer_index]
            recurrent_regularizer = None
            if self._recurrent_regularizer is not None:
                recurrent_regularizer = self._recurrent_regularizer[layer_index]
            kernel_regularizer = None
            if self._kernel_regularizer is not None:
                kernel_regularizer = self._kernel_regularizer[layer_index]
            bias_regularizer = None
            if self._bias_regularizer is not None:
                bias_regularizer = self._bias_regularizer[layer_index]
            activity_regularizer = None
            if self._activity_regularizer is not None:
                activity_regularizer = self._activity_regularizer[layer_index]

            # Construct either LSTM or GRU
            recurrent_layer = constructor(
                name=f"decoder_{unit_type.lower()}{len(self._unit_types) - layer_index + 1}",
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_initializer=self._bias_initializer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                recurrent_initializer=self._recurrent_initializer,
                recurrent_regularizer=recurrent_regularizer,
                dropout=self._recurrent_units_dropout,
                recurrent_dropout=self._recurrent_dropout,
                return_sequences=True,
                go_backwards=self._go_backwards
            )
            decoder_layers.append(recurrent_layer)

        return decoder_layers

    def get_config(self) -> dict[str, Union[None, list[Optional[dict[str, Any]]], tuple, int]]:
        sup_config = super(RecurrentAutoEncoder, self).get_config()
        del sup_config[AutoEncoder.ENCODER_CONFIG]
        del sup_config[AutoEncoder.DECODER_CONFIG]
        config = {
            "input_shape": self.input_shape,
            "unit_types": self._unit_types,
            "recurrent_units": self._recurrent_units,
            "latent_space_dim": self.latent_space_dim,
            "activations": self._activations,
            "recurrent_activations": self._recurrent_activations,
            "bottleneck_unit_type": self._bottleneck_unit_type,
            "bottleneck_returns_sequences": self._bottleneck_returns_sequences,
            "bottleneck_activation": self._bottleneck_activation,
            "bottleneck_recurrent_activation": self._bottleneck_recurrent_activation,
            "bottleneck_recurrent_regularizer": self._bottleneck_recurrent_regularizer,
            "bottleneck_kernel_regularizer": self._bottleneck_kernel_regularizer,
            "bottleneck_bias_regularizer": self._bottleneck_bias_regularizer,
            "bottleneck_activity_regularizer": self._bottleneck_activity_regularizer,
            "recurrent_units_dropout": self._recurrent_units_dropout,
            "recurrent_dropout": self._recurrent_dropout,
            "recurrent_initializer": self._recurrent_initializer,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "recurrent_regularizer": self._recurrent_regularizer,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "activity_regularizer": self._activity_regularizer,
            "go_backwards": self._go_backwards,
            **sup_config
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
