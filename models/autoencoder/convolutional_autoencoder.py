from typing import Iterable, final, Optional
from autoencoder import AutoEncoder
from keras.layers import MaxPooling1D, UpSampling1D, Conv1D, Layer, BatchNormalization, AveragePooling1D, Flatten, \
    Dense, Dropout, Conv1DTranspose, Reshape
import keras.backend as K

AVG_POOL: final = "AVG"
MAX_POOL: final = "MAX_POOL"


class FlattenDenseLayer(Layer):
    """
    This class represents a simple neural network layer composed of a Flatten layer followed by fully-connected (Dense)
    layer.
    """

    # TODO: document this copying doc from keras Dense and Flatten docs
    def __init__(self, output_dim: int, flatten_data_format: Optional[str] = None, name: Optional[str] = None,
                 activation=None, use_bias: bool = True, kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros', dropout: float = 0.0, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):

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
        self._dropout = Dropout(rate=dropout)
        self._shape_before_flatten = None

    def call(self, inputs, *args, **kwargs):
        self._shape_before_flatten = K.int_shape(inputs)[1:]  # Input shape before flatten excluding batch size
        flattened = self._flatten_layer(inputs)
        dense_output = self._dense(flattened)
        return self._dropout(dense_output)

    def build(self, input_shape):
        self._flatten_layer.build(input_shape)
        self._dense.build(input_shape)
        super(FlattenDenseLayer, self).build(input_shape)

    @property
    def shape_before_flatten(self) -> Optional[tuple[int]]:
        return self._shape_before_flatten

    @property
    def units(self) -> int:
        return self._dense.units


class Convolutional1DAutoEncoder(AutoEncoder):
    """
    This class represents a 1-dimensional convolutional autoencoder model.
    """

    def __init__(self, n_features: int, conv_filters: list[int], conv_kernels_size: list[int], conv_strides: list[int],
                 latent_space_dim: int, conv_pools: list[Optional[int]] = None, dropout_conv: float = 0.0,
                 dropout_dense: float = 0.0, pool_type: str = AVG_POOL, activation: str = 'relu',
                 ignore_first_convolutional_decoder: bool = False, input_shape: Optional[Iterable[int]] = None):
        # TODO: add checks, error rising and documentation

        # Setup instance variables
        self._conv_filters = conv_filters
        self._conv_kernels_size = conv_kernels_size
        self._conv_strides = conv_strides
        self._conv_pools = conv_pools
        self._pool_type = pool_type
        self._n_convolution_layers = len(conv_filters)
        self._activation = activation
        self._shape_before_bottleneck = None
        self._dropout_conv = dropout_conv
        self._dropout_dense = dropout_dense

        # Build convolutional layer blocks for encoder, decoder and bottleneck
        encoder_conv_blocks = self._build_encoder_conv_blocks()
        bottleneck = self._build_bottleneck(latent_space_dim)
        decoder_conv_blocks = self._build_decoder(bottleneck, ignore_first_convolutional_decoder)

        # Call the parent constructor passing the created layers to it
        super(Convolutional1DAutoEncoder, self).__init__(
            n_features=n_features,
            encoder_layers=encoder_conv_blocks,
            bottleneck=bottleneck,
            decoder_layers=decoder_conv_blocks,
            outputs_sequences=False,
            input_shape=input_shape
        )

    def _build_encoder_conv_blocks(self) -> Iterable[Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization]:
        """
        Builds all encoder convolutional blocks, composed of a Conv1D (ReLU activation function by default) layer,
        a BatchNormalization layer and a max/average pooling layer (if required).

        :return list of created convolutional encoder blocks.
        """
        encoder_conv_blocks = []

        # For each conv layer index, build a new convolutional block and add it to the list
        for layer_index in range(0, self._n_convolution_layers):
            encoder_block = self._build_encoder_conv_block(layer_index)
            encoder_conv_blocks.extend(encoder_block)

        return encoder_conv_blocks

    def _build_encoder_conv_block(self, layer_index: int) -> Iterable[Conv1D, MaxPooling1D, AveragePooling1D,
                                                                      BatchNormalization, Dropout]:
        """
        Builds a new convolutional block, composed of a Conv1D (ReLU activation function by default) layer, a
        BatchNormalization layer and a max/average pooling layer (if required).

        :param layer_index: convolutional block index (in the encoder model)
        :return: created convolutional block.
        """
        conv_block = []

        # Create convolutional and batch normalization layers
        conv_layer = Conv1D(
            filters=self._conv_filters[layer_index],
            kernel_size=self._conv_kernels_size[layer_index],
            strides=self._conv_strides[layer_index],
            padding="same",
            activation=self._activation,
            name=f"conv_layer_encoder{layer_index}"
        )
        batch_normalization_layer = BatchNormalization(name=f"bn_layer_encoder{layer_index}")

        # Add them to the output
        conv_block.append(conv_layer)
        conv_block.append(batch_normalization_layer)

        # Add dropout layer if conv dropout is not 0
        if self.dropout_layer != 0:
            conv_dropout_layer = Dropout(self._dropout_conv)
            conv_block.append(conv_dropout_layer)

        # Create pooling layer and add it to the output (if required)
        if self._conv_pools is not None and self._conv_pools[layer_index] is not None:
            pool_layer = None
            if self._pool_type == AVG_POOL:
                pool_layer = AveragePooling1D(self._conv_pools[layer_index], padding='same')
            elif self._pool_type == MAX_POOL:
                pool_layer = MaxPooling1D(self._conv_pools[layer_index], padding='same')
            conv_block.append(pool_layer)

        return conv_block

    def _build_bottleneck(self, latent_space_dim: int) -> Layer:
        """
        Build the bottleneck layers, which consists of a FlattenDenseLayer.

        :param latent_space_dim: latent space dimension.
        :return: created bottleneck FlattenDenseLayer with given latent space dimension.
        """
        flatten_dense_layer = FlattenDenseLayer(latent_space_dim, name="encoder_output", dropout=self._dropout_dense)
        return flatten_dense_layer

    # TODO: write build decoder functions
    def _build_decoder(self, bottleneck: FlattenDenseLayer, ignore_first_convolutional_decoder: bool) \
            -> Iterable[Dense, Dropout, Reshape, Conv1DTranspose, UpSampling1D]:
        decoder_layers = []

        # Add decoder dense layer (and corresponding dropout if given) to the decoder
        decoder_dense = Dense(units=bottleneck.units, name='decoder_dense')
        decoder_layers.append(decoder_dense)

        if self._dropout_dense != 0:
            decoder_dropout = Dropout(rate=self._dropout_dense)
            decoder_layers.append(decoder_dropout)

        # Add reshape layers (to reshape from flattened to original, saved in the bottleneck)
        decoder_reshape_layer = Reshape(bottleneck.shape_before_flatten)
        decoder_layers.append(decoder_reshape_layer)

        # Add all convolutional transpose layers
        decoder_conv_transpose_layers = self._build_decoder_conv_transpose_blocks(ignore_first_convolutional_decoder)

        return decoder_layers

    def _build_conv_transpose_blocks(self, ignore_first_convolutional_decoder: bool) -> Iterable[Conv1DTranspose,
                                                                                                 UpSampling1D, Dropout]:
        decoder_conv_transpose_blocks = []

        # For each conv layer index in reverse order (ignoring the first convolutional if parameter is given), build a
        # new de-convolutional block and add it to the lis
        conv_starting_index = 1 if ignore_first_convolutional_decoder else 0
        for layer_index in reversed(range(conv_starting_index, self._n_convolution_layers)):
            decoder_conv_transpose_block = self._build_decoder_deconv_block(layer_index)
            decoder_conv_transpose_blocks.extend(decoder_conv_transpose_block)

        return decoder_conv_transpose_blocks

    def _build_decoder_deconv_block(self, layer_index: int):
        deconv_block = []

        # Create convolutional and batch normalization layers
        deconv_layer = Conv1DTranspose(
            filters=self._conv_filters[layer_index],
            kernel_size=self._conv_kernels_size[layer_index],
            strides=self._conv_strides[layer_index],
            padding="same",
            activation=self._activation,
            name=f"deconv_layer_decoder{layer_index}"
        )
        batch_normalization_layer = BatchNormalization(name=f"bn_layer_decoder{layer_index}")

        # Add them to the output
        deconv_block.append(deconv_layer)
        deconv_block.append(batch_normalization_layer)

        # Add dropout layer if conv dropout is not 0
        if self.dropout_layer != 0:
            conv_dropout_layer = Dropout(self._dropout_conv)
            deconv_block.append(conv_dropout_layer)

        # Create up sampling layer (to reverse the pool layer) and add it to the output (if required)
        if self._conv_pools is not None and self._conv_pools[layer_index] is not None:
            up_sampling_layer = UpSampling1D(self._conv_pools[layer_index], padding='same')
            deconv_block.append(up_sampling_layer)

        return deconv_block





