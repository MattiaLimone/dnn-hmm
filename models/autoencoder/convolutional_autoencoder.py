from typing import Iterable, final, Optional, Union
import numpy as np
from autoencoder import AutoEncoder
from keras.layers import MaxPooling1D, UpSampling1D, Conv1D, Layer, BatchNormalization, AveragePooling1D, Flatten, \
    Dense, Dropout, Conv1DTranspose, Reshape
import keras.backend as K

AVG_POOL: final = "AVG"
MAX_POOL: final = "MAX_POOL"
_RANDOM_SEED: final = None


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
        self._dropout = Dropout(rate=dropout, seed=_RANDOM_SEED)

    def call(self, inputs, *args, **kwargs):
        flattened = self._flatten_layer(inputs)
        dense_output = self._dense(flattened)
        return self._dropout(dense_output)

    def build(self, input_shape):
        self._flatten_layer.build(input_shape)
        flattened_shape = self._flatten_layer.compute_output_shape(input_shape)
        self._dense.build(flattened_shape)
        dense_output_shape = self._dense.compute_output_shape(flattened_shape)
        self._dropout.build(dense_output_shape)
        super(FlattenDenseLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        flattened_input_shape = self._flatten_layer.compute_output_shape(input_shape)
        return self._dense.compute_output_shape(flattened_input_shape)

    @property
    def units(self) -> int:
        return self._dense.units


class Convolutional1DAutoEncoder(AutoEncoder):
    """
    This class represents a 1-dimensional convolutional autoencoder model.
    """

    def __init__(self, input_shape: tuple[int, ...], conv_filters: list[int], conv_kernels_size: list[int],
                 conv_strides: list[int], latent_space_dim: int, conv_pools: list[Optional[int]] = None,
                 dropout_conv: float = 0.0, dropout_dense: float = 0.0, pool_type: str = AVG_POOL,
                 activation: str = 'relu', ignore_first_convolutional_decoder: bool = False):
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
        # Final output shape of the convolutional encoder blocks shape is the shape before the bottleneck flattening
        encoder_conv_blocks, conv_output_shape = self._build_encoder_conv_blocks(input_shape)
        bottleneck, bottleneck_output_shape = self._build_bottleneck(conv_output_shape, latent_space_dim)
        decoder_conv_blocks = self._build_decoder(conv_output_shape, ignore_first_convolutional_decoder)

        # Call the parent constructor passing the created layers to it
        super(Convolutional1DAutoEncoder, self).__init__(
            n_features=input_shape[-1],
            encoder_layers=encoder_conv_blocks,
            bottleneck=bottleneck,
            decoder_layers=decoder_conv_blocks,
            outputs_sequences=False,
            input_shape=input_shape
        )

    def _build_encoder_conv_blocks(self, input_shape: tuple[int, ...]) -> \
            (Iterable[Union[Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization]], tuple[int, ...]):
        """
        Builds all encoder convolutional blocks, composed of a Conv1D (ReLU activation function by default) layer,
        a BatchNormalization layer and a max/average pooling layer (if required).

        :param input_shape: tuple of integers representing the input shape.

        :return list of created convolutional encoder blocks, and the output shape of the last convolutional block.
        """
        encoder_conv_blocks = []
        output_shape = input_shape

        # For each conv layer index, build a new convolutional block and add it to the list
        for layer_index in range(0, self._n_convolution_layers):
            encoder_block, output_shape = self._build_encoder_conv_block(layer_index, input_shape=output_shape)
            encoder_conv_blocks.extend(encoder_block)

        return encoder_conv_blocks, output_shape

    def _build_encoder_conv_block(self, layer_index: int, input_shape: tuple[int, ...]) -> \
            (Iterable[Union[Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Dropout]], tuple[int, ...]):
        """
        Builds a new convolutional block, composed of a Conv1D (ReLU activation function by default) layer, a
        BatchNormalization layer and a max/average pooling layer (if required).

        :param layer_index: convolutional block index (in the encoder model)
        :param input_shape: tuple of integers representing the input shape.
        :return: created convolutional block and block output shape.
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
        output_shape = conv_layer.compute_output_shape(input_shape=input_shape)

        batch_normalization_layer = BatchNormalization(name=f"bn_layer_encoder{layer_index}")
        output_shape = batch_normalization_layer.compute_output_shape(input_shape=output_shape)

        # Add them to the output
        conv_block.append(conv_layer)
        conv_block.append(batch_normalization_layer)

        # Add dropout layer if conv dropout is not 0
        if self._dropout_conv != 0:
            conv_dropout_layer = Dropout(rate=self._dropout_conv, seed=_RANDOM_SEED)
            output_shape = conv_dropout_layer.compute_output_shape(output_shape)
            conv_block.append(conv_dropout_layer)

        # Create pooling layer and add it to the output (if required)
        if self._conv_pools is not None and self._conv_pools[layer_index] is not None:
            pool_layer = None
            if self._pool_type == AVG_POOL:
                pool_layer = AveragePooling1D(self._conv_pools[layer_index], padding='same')
            elif self._pool_type == MAX_POOL:
                pool_layer = MaxPooling1D(self._conv_pools[layer_index], padding='same')
            output_shape = pool_layer.compute_output_shape(output_shape)
            conv_block.append(pool_layer)

        return conv_block, output_shape

    def _build_bottleneck(self, conv_output_shape: tuple[int, ...], latent_space_dim: int) -> (Layer, tuple[int, ...]):
        """
        Build the bottleneck layers, which consists of a FlattenDenseLayer.

        :param conv_output_shape: final output shape of the convolutional blocks.
        :param latent_space_dim: latent space dimension.
        :return: created bottleneck FlattenDenseLayer with given latent space dimension.
        """
        flatten_dense_layer = FlattenDenseLayer(latent_space_dim, name="encoder_output", dropout=self._dropout_dense)
        bottleneck_output_shape = flatten_dense_layer.compute_output_shape(conv_output_shape)
        return flatten_dense_layer, bottleneck_output_shape

    # TODO: write build decoder functions
    def _build_decoder(self, conv_output_shape: tuple[int, ...], ignore_first_convolutional_decoder: bool = False) \
            -> Iterable[Union[Dense, Dropout, Reshape, Conv1DTranspose, UpSampling1D]]:
        decoder_layers = []

        # Add decoder dense layer (and corresponding dropout if given) to the decoder, which output is the flattened
        # final shape of the convolutional block (batch size excluded)
        decoder_dense = Dense(units=np.prod(conv_output_shape[1:]), name='decoder_dense')
        decoder_layers.append(decoder_dense)

        # Add dropout layer if required
        if self._dropout_dense != 0:
            decoder_dropout = Dropout(rate=self._dropout_dense, seed=_RANDOM_SEED)
            decoder_layers.append(decoder_dropout)

        # Add reshape layers to invert the Flatten layers
        decoder_reshape_layer = Reshape(target_shape=conv_output_shape[1:])
        decoder_layers.append(decoder_reshape_layer)

        # Add all convolutional transpose layers
        decoder_conv_transpose_layers = self._build_decoder_conv_transpose_blocks(ignore_first_convolutional_decoder)
        decoder_layers.extend(decoder_conv_transpose_layers)

        return decoder_layers

    def _build_decoder_conv_transpose_blocks(self, ignore_first_convolutional_decoder: bool) -> \
            Iterable[Union[Conv1DTranspose, UpSampling1D, Dropout]]:
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

        # Create up sampling layer (to reverse the pool layer) and add it to the output (if required)
        if self._conv_pools is not None and self._conv_pools[layer_index] is not None:
            up_sampling_layer = UpSampling1D(self._conv_pools[layer_index])
            deconv_block.append(up_sampling_layer)

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
        if self._dropout_conv != 0:
            conv_dropout_layer = Dropout(self._dropout_conv, seed=_RANDOM_SEED)
            deconv_block.append(conv_dropout_layer)

        return deconv_block





