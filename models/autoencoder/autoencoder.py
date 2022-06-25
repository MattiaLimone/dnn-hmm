import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, TimeDistributed, Layer, InputLayer, BatchNormalization, Flatten, Dropout, Reshape
from typing import final, Optional, Union, Any, Iterable


ENCODER_MODEL_NAME: final = "Encoder"
DECODER_MODEL_NAME: final = "Decoder"
_DECODER_LAYER_DEFAULT_POSTFIX = "_decoder"
_RANDOM_SEED: final = None


class FlattenDenseLayer(Layer):
    """
    This class represents a simple neural network layer composed of a Flatten layer followed by fully-connected (Dense)
    layer.
    """

    def __init__(self, output_dim: int, flatten_data_format: Optional[str] = None, name: Optional[str] = None,
                 activation=None, use_bias: bool = True, kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros', dropout: float = 0.0, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        """
        Constructor. Instantiates a Flatten layer followed by a Dense layer that represents the bottleneck.

        :param output_dim: An integer. Dimension of the output.
        :param flatten_data_format: A string, one of channels_last (default) or channels_first. The ordering of the
            dimensions in the inputs. channels_last corresponds to inputs with shape (batch, ..., channels) while
            channels_first corresponds to inputs with shape (batch, channels, ...). It defaults to the image_data_format
            value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be
            "channels_last".
        :param name: A string. The name of the layer.
        :param activation: Activation function to use. If you don't specify anything, no activation is applied.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param kernel_initializer: Initializer for the kernel weights matrix.
        :param bias_initializer: Initializer for the bias vector.
        :param dropout: Float between 0 and 1. Fraction of the input units to drop.
        :param kernel_regularizer: Regularizer function applied to the kernel weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        :param activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
        :param kernel_constraint: Constraint function applied to the kernel weights matrix.
        :param bias_constraint: Constraint function applied to the bias vector.
        :param kwargs: Additional parameters inherited from Flatten or Dense layer.
        """

        super(FlattenDenseLayer, self).__init__(trainable=True, name=name)

        self._flatten_layer = Flatten(data_format=flatten_data_format)
        self._dense = Dense(
            units=output_dim,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._dropout = Dropout(rate=dropout, seed=_RANDOM_SEED)

    def call(self, inputs, *args, **kwargs):
        """
        Calls the model on new inputs and returns the outputs as tensors, flattening the input tensors on a single
        dimension, and passing them to the dense layer.

        :param inputs: Input tensor, or dict/list/tuple of input tensors.
        :param args: Additional positional arguments. May contain tensors, although this is not recommended.
        :param kwargs: Additional keyword arguments. May contain tensors, although this is not recommended.
        :return: A tensor or list/tuple of tensors containing the flattened inputs passed through a dense layer.
        """
        flattened = self._flatten_layer(inputs)
        dense_output = self._dense(flattened)
        return self._dropout(dense_output)

    def build(self, input_shape):
        """
        Creates the variables of the layer

        :param input_shape: Instance of `TensorShape`, or list of instances of `TensorShape` if the layer expects a list
            of inputs (one instance per input).
        """
        self._flatten_layer.build(input_shape)
        flattened_shape = self._flatten_layer.compute_output_shape(input_shape)
        self._dense.build(flattened_shape)
        dense_output_shape = self._dense.compute_output_shape(flattened_shape)
        self._dropout.build(dense_output_shape)
        super(FlattenDenseLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        :param input_shape: shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions, instead of an integer.
        :return: An input shape tuple.
        """
        flattened_input_shape = self._flatten_layer.compute_output_shape(input_shape)
        return self._dense.compute_output_shape(flattened_input_shape)

    @property
    def units(self) -> int:
        """
        Returns the units number of the dense layer

        :return: An integer representing the number of units of the dense layer
        """
        return self._dense.units


class AutoEncoder(Model):
    """
    This class represents a generic autoencoder model, that can be constructed with any keras layer.
    """

    def __init__(self, input_shape: tuple[int, ...], encoder_layers: Iterable[Layer], bottleneck: Layer,
                 decoder_layers: Optional[Iterable[Layer]] = None, add_last_dense_block: bool = True,
                 do_batch_norm: bool = False, outputs_sequences: bool = False, last_layer_activation=None):
        """
        Constructor. Instantiates a new autoencoder with the given encoder and decoder layers and builds it, if input
        shape is given.

        :param input_shape: a integer tuple representing the input shape the model will be build from (batch_size, dim0,
            dim1, ..., dimN).
        :param encoder_layers: an iterable containing the encoder layers (no InputLayer must be given, or ValueError
                               will be raised).
        :param bottleneck: bottleneck layer which outputs the representation of the input vector in the latent space.
        :param decoder_layers: an iterable containing the decoder layers (no InputLayer must be given, or ValueError
                               will be raised); by default, this is None since the autoencoder structure is assumed to
                               be symmetrical (hence encoder layers are copied in reverse order in decoder layers).
        :param add_last_dense_block: whether to add last dense block after decoder layers.
        :param do_batch_norm: whether or not to add a batch normalization layer before the output layer of the decoder.
        :param outputs_sequences: a boolean indicating whether or not the output of the network should be a sequence; if
            true, input_shape[1] will be used as timesteps, and the rest of dimensions as output shape.
        :param last_layer_activation: activation function applied to the output layer, if None is given, then a linear
                                      activation function is applied (a(x) = x).

        :raises ValueError: if given n_features is less than 1, if input_shape last element does not coincide with
                            n_features or if one of the layers contained in encoder_layers or decoder_layers is an
                            instance of InputLayer.
        """
        if input_shape[-1] < 1:
            raise ValueError("Feature number must be strictly positive")

        super(AutoEncoder, self).__init__()
        self._encoder = Sequential(name=ENCODER_MODEL_NAME)
        self._decoder = Sequential(name=DECODER_MODEL_NAME)
        self._input_shape = input_shape

        # If autoencoder must be symmetrical
        if decoder_layers is None:

            # Add all given layers to the encoder model
            for layer in encoder_layers:

                # Raise error if one of the given layers is an InputLayer
                if isinstance(layer, InputLayer):
                    raise ValueError("Given layers must not be InputLayer instances")

                self._encoder.add(layer)

            # Add bottleneck
            self._encoder.add(bottleneck)

            # Add all given layers in reverse order to the decoder model
            reverse_layers = list(encoder_layers)
            reverse_layers.reverse()

            for layer in reverse_layers:
                layer_config = layer.get_config()
                layer_name = layer_config["name"] + _DECODER_LAYER_DEFAULT_POSTFIX
                layer_config["name"] = layer_name
                cloned_layer = type(layer).from_config(layer_config)
                self._decoder.add(cloned_layer)

        # If decoder layers are given explicitly
        else:
            # Add encoder layers
            for layer in encoder_layers:

                # Raise error if one of the given layers is an InputLayer
                if isinstance(layer, InputLayer):
                    raise ValueError("Given layers must not be InputLayer instances")

                self._encoder.add(layer)

            # Add bottleneck
            self._encoder.add(bottleneck)

            # Add decoder layers
            for layer in decoder_layers:

                # Raise error if one of the given layers is an InputLayer
                if isinstance(layer, InputLayer):
                    raise ValueError("Given layers must not be InputLayer instances")
                self._decoder.add(layer)

        # Compute encoder output shape and decoder (temporary) output shape (batch_size, other dimensions...)
        encoder_output_shape = self._encoder.compute_output_shape(input_shape)
        self._latent_space_dim = bottleneck.units  # number of features in latent space
        decoder_output_shape = self._decoder.compute_output_shape(encoder_output_shape)

        if add_last_dense_block:

            # Add batch normalization layer before the output layer, if required
            if do_batch_norm:
                self._decoder.add(BatchNormalization(name="output_batch_normalization"))

            # Add last layer that has the same size as the input of the network
            if outputs_sequences:
                # Flatten on all dimensions except for the batch size and timesteps
                timesteps = input_shape[1]
                flattened_input_shape = np.product(input_shape[2:])
                decoder_output_shape_flattened = np.product(decoder_output_shape[2:])
                self._decoder.add(Reshape((timesteps, decoder_output_shape_flattened), name="pre_output_flatten"))

                # Add time distributed dense output layer
                self._decoder.add(
                    TimeDistributed(Dense(flattened_input_shape, activation=last_layer_activation), name="output_dense")
                )

                # Reshape output to correct tensor shape (batch_size, timesteps, other dimensions...)
                self._decoder.add(Reshape((timesteps, ) + input_shape[2:], name="output_reshape"))

            else:
                # Flatten on all dimensions except for the batch size
                flattened_input_shape = np.product(input_shape[1:])
                decoder_output_shape_flattened = np.product(decoder_output_shape[1:])
                self._decoder.add(Reshape((decoder_output_shape_flattened, ), name="pre_output_flatten"))

                # Add dense output layer
                self._decoder.add(
                    Dense(flattened_input_shape, activation=last_layer_activation, name="output_dense")
                )

                # Reshape output to correct tensor shape (batch_size, other dimensions...)
                self._decoder.add(Reshape(input_shape[1:], name="output_reshape"))

        decoder_output_shape = self._decoder.compute_output_shape(encoder_output_shape)

        if decoder_output_shape != input_shape:
            raise ValueError("Input shape and output shape must match exactly")

        # Build the model
        self.build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """
        Calls the model on new inputs and returns the outputs as tensors, encoding the input tensor in the latent space
        and trying to reconstruct the input it from this latent representation.

        :param inputs: input tensor, or dict/list/tuple of input tensors.
        :param training: boolean or boolean scalar tensor, indicating whether to run the autoencoder in training mode
                         or inference mode.
        :param mask: a mask or list of masks. A mask can be either a boolean tensor or None (no mask).
                     For more details, check the guide
                     [here](https://www.tensorflow.org/guide/keras/masking_and_padding).
        :return: tensor representing the reconstructed input.
        """
        encoded = self._encoder(inputs)
        decoded = self._decoder(encoded)
        return decoded

    def get_config(self) -> dict[str, Union[None, list[Optional[dict[str, Any]]], tuple, int]]:
        config_dict = {
            "latent_space_dim": self._latent_space_dim,
            **self._encoder.get_config(),
            **self._decoder.get_config()
        }
        return config_dict

    def fit(self, x=None, y=None, batch_size: Optional[int] = None, epochs: int = 1, verbose: str = 'auto',
            callbacks=None, validation_split: float = 0.0, validation_data=None, shuffle: bool = True,
            class_weight=None, sample_weight=None, initial_epoch: int = 0, steps_per_epoch=None, validation_steps=None,
            validation_batch_size: Optional[int] = None, validation_freq: int = 1, max_queue_size: int = 10,
            workers: int = 1, use_multiprocessing: bool = False):

        """Trains the model for a fixed number of epochs (iterations on a dataset).

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
          - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
            callable that takes a single argument of type
            `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
            `DatasetCreator` should be used when users prefer to specify the
            per-replica batching and sharding logic for the `Dataset`.
            See `tf.keras.utils.experimental.DatasetCreator` doc for more
            information.
          A more detailed description of unpacking behavior for iterator types
          (Dataset, generator, Sequence) is given below. If using
          `tf.distribute.experimental.ParameterServerStrategy`, only
          `DatasetCreator` type is supported for `x`.
        y: Target data. Must be `None` in autoencoder context, since the training data is the same as the label data.
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided
            (unless the `steps_per_epoch` flag is set to
            something other than None).
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 'auto', 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so verbose=2 is
            recommended when not running interactively (eg, in a production
            environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
            and `tf.keras.callbacks.History` callbacks are created automatically
            and need not be passed into `model.fit`.
            `tf.keras.callbacks.ProgbarLogger` is created or not based on
            `verbose` argument to `model.fit`.
            Callbacks with batch-level calls are currently unsupported with
            `tf.distribute.experimental.ParameterServerStrategy`, and users are
            advised to implement epoch-level calls instead with an appropriate
            `steps_per_epoch` value.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
           `keras.utils.Sequence` instance.
            `validation_split` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Thus, note the fact
            that the validation loss of data provided using `validation_split`
            or `validation_data` is not affected by regularization layers like
            noise and dropout.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
              - A tuple `(x_val, y_val, val_sample_weights)` of NumPy arrays.
              - A `tf.data.Dataset`.
              - A Python generator or `keras.utils.Sequence` returning
              `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            `validation_data` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch'). This argument is ignored
            when `x` is a generator or an object of tf.data.Dataset.
            'batch' is a special option for dealing
            with the limitations of HDF5 data; it shuffles in batch-sized
            chunks. Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample. This
            argument is not supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            When passing an infinitely repeating dataset, you must specify the
            `steps_per_epoch` argument. If `steps_per_epoch=-1` the training
            will run indefinitely with an infinitely repeating dataset.
            This argument is not supported with array inputs.
            When using `tf.distribute.experimental.ParameterServerStrategy`:
              * `steps_per_epoch=None` is not supported.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of an infinitely repeated dataset, it will run into an
            infinite loop. If 'validation_steps' is specified and only part of
            the dataset will be consumed, the evaluation will start from the
            beginning of the dataset at each epoch. This ensures that the same
            validation samples are used every time.
        validation_batch_size: Integer or `None`.
            Number of samples per validation batch.
            If unspecified, will default to `batch_size`.
            Do not specify the `validation_batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

    Unpacking behavior for iterator-like inputs:
        A common pattern is to pass a tf.data.Dataset, generator, or
      tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
      yield not only features (x) but optionally targets (y) and sample weights.
      Keras requires that the output of such iterator-likes be unambiguous. The
      iterator should return a tuple of length 1, 2, or 3, where the optional
      second and third elements will be used for y and sample_weight
      respectively. Any other type provided will be wrapped in a length one
      tuple, effectively treating everything as 'x'. When yielding dicts, they
      should still adhere to the top-level tuple structure.
      e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
      features, targets, and weights from the keys of a single dict.
        A notable unsupported data type is the namedtuple. The reason is that
      it behaves like both an ordered datatype (tuple) and a mapping
      datatype (dict). So given a namedtuple of the form:
          `namedtuple("example_tuple", ["y", "x"])`
      it is ambiguous whether to reverse the order of the elements when
      interpreting the value. Even worse is a tuple of the form:
          `namedtuple("other_tuple", ["x", "y", "z"])`
      where it is unclear if the tuple was intended to be unpacked into x, y,
      and sample_weight or passed through as a single element to `x`. As a
      result the data processing code will simply raise a ValueError if it
      encounters a namedtuple. (Along with instructions to remedy the issue.)

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: 1. If the model was never compiled or,
        2. If `model.fit` is  wrapped in `tf.function`.

        ValueError: In case of mismatch between the provided input data
            and what the model expects or when the input data is empty.

        ValueError: If target data tensor is given.
    """
        if y is not None:
            raise ValueError("AutoEncoder must fit just on input data.")

        return super(AutoEncoder, self).fit(
            x=x,
            y=x,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size, workers=workers,
            use_multiprocessing=use_multiprocessing
        )

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
        super(AutoEncoder, self).summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable
        )

    @property
    def latent_space_dim(self) -> int:
        return self._latent_space_dim
