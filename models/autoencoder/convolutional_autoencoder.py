from typing import Iterable, final, Optional, Union
import numpy as np
from autoencoder import AutoEncoder
from keras.layers import MaxPooling1D, UpSampling1D, Conv1D, Layer, BatchNormalization, AveragePooling1D, Flatten, \
    Dense, Dropout, Conv1DTranspose, Reshape
from autoencoder import FlattenDenseLayer, _RANDOM_SEED

AVG_POOL: final = "AVG"
MAX_POOL: final = "MAX"


class Convolutional1DAutoEncoder(AutoEncoder):
    """
    This class represents a 1-dimensional convolutional autoencoder model.
    """

    def __init__(self, input_shape: tuple[int, ...], conv_filters: list[int], conv_kernels_size: list[int],
                 conv_strides: list[int], latent_space_dim: int, conv_pools: list[Optional[int]] = None,
                 dropout_conv: float = 0.0, dropout_dense: float = 0.0, pool_type: str = AVG_POOL,
                 activation='relu', ignore_first_convolutional_decoder: bool = False, do_batch_norm: bool = False):

        """
        Constructor. Instantiates a new convolutional autoencoder with the given encoder and decoder layers and builds
        it, if input shape is given.

        :param input_shape: A tuple/list of integer. The shape format of the input.
        :param conv_filters: Integer, the dimensionality of the output space for each layer.
        :param conv_kernels_size: An integer or tuple/list of a single integer, specifying the length of the 1D
            convolution window.
        :param conv_strides: An integer or tuple/list of a single integer, specifying the stride length of the
            convolution.
        :param latent_space_dim: An integer. Dimensionality of the bottleneck.
        :param conv_pools: An integer or tuple/list of a single integer, specifying the dimensionality of the pooling
            layers.
        :param dropout_conv: Float between 0 and 1. Fraction of the input units to drop after Convolutional Layer.
        :param dropout_dense: Float between 0 and 1. Fraction of the input units to drop after Dense Layer.
        :param pool_type: A string. Either AVG_POOL" or MAX_POOL to use an Average Pooling layer or a Max Pooling
            layer.
        :param activation: Activation function to use. If you don't specify anything, no activation is applied.
        :param do_batch_norm: whether or not to add a batch normalization layer before the output layer of the decoder.
        :param ignore_first_convolutional_decoder: A boolean. If true first convolutional layer of the encoder will not
            be added to the decoder as a deconvolutional layer.
        """
        if not input_shape and input_shape[-1] < 1:
            raise ValueError('Feature number must be strictly positive. '
                             f'Received input_shape={input_shape}.')
        if not conv_filters and len(conv_filters) <= 0:
            raise ValueError('Invalid value for argument `conv_filters`. Expected a strictly positive value. '
                             f'Received conv_filters={conv_filters}.')
        if not conv_kernels_size and len(conv_kernels_size) <= 0:
            raise ValueError('Invalid value for argument `conv_kernels_size`. Expected a strictly positive value. '
                             f'Received conv_kernels_size={conv_kernels_size}.')
        if not conv_strides and len(conv_strides) <= 0:
            raise ValueError('Invalid value for argument `conv_strides`. Expected a strictly positive value. '
                             f'Received conv_kernels_size={conv_strides}.')
        if not latent_space_dim and latent_space_dim <= 0:
            raise ValueError('Invalid value for argument `latent_space_dim`. Expected a strictly positive value. '
                             f'Received latent_space_dim={latent_space_dim}.')
        if not dropout_conv and dropout_conv < 0:
            raise ValueError('Invalid value for argument `dropout_conv`. Expected a positive value. '
                             f'Received dropout_conv={dropout_conv}.')
        if not dropout_dense and dropout_dense < 0:
            raise ValueError('Invalid value for argument `dropout_dense`. Expected a positive value. '
                             f'Received dropout_dense={dropout_dense}.')
        if len(conv_filters) != len(conv_kernels_size) or len(conv_filters) != len(conv_strides):
            raise ValueError('Invalid value for argument `conv_filters`,`conv_kernels_size` or `conv_stride. Same '
                             'dimension expected.'
                             f'\nReceived conv_filters dimension={len(conv_filters)}.'
                             f'\nReceived conv_kernels_size dimension={len(conv_kernels_size)}.'
                             f'\nReceived conv_strides dimension={len(conv_strides)}.')

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
            input_shape=input_shape,
            encoder_layers=encoder_conv_blocks,
            bottleneck=bottleneck,
            decoder_layers=decoder_conv_blocks,
            outputs_sequences=False,
            do_batch_norm=do_batch_norm
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
        flatten_dense_layer = FlattenDenseLayer(
            latent_space_dim,
            name="encoder_output",
            activation='relu',
            dropout=self._dropout_dense
        )
        bottleneck_output_shape = flatten_dense_layer.compute_output_shape(conv_output_shape)
        return flatten_dense_layer, bottleneck_output_shape

    def _build_decoder(self, conv_output_shape: tuple[int, ...], ignore_first_convolutional_decoder: bool = False) \
            -> Iterable[Union[Dense, Dropout, Reshape, Conv1DTranspose, UpSampling1D]]:
        """
        Build the decoder layers, which consists of the symmetrical architecture of the decoder.

        :param conv_output_shape: final output shape of the bottleneck layer.
        :param ignore_first_convolutional_decoder: A boolean. If true first layer of decoder will not be added to the
            model.
        :return: created DecoderLayer with given output shape
        """
        decoder_layers = []

        # Add decoder dense layer (and corresponding dropout if given) to the decoder, which output is the flattened
        # final shape of the convolutional block (batch size excluded)
        decoder_dense = Dense(units=np.prod(conv_output_shape[1:]), name='decoder_dense', activation='relu')
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
        """
        Build the upsampling layers of the decoder to reconstruct the input in the decoder layer.

        :param ignore_first_convolutional_decoder: A boolean. If true first layer of decoder will not be added to the
            model.
        :return: created transpose block for each convolutional layer
        """
        decoder_conv_transpose_blocks = []

        # For each conv layer index in reverse order (ignoring the first convolutional if parameter is given), build a
        # new de-convolutional block and add it to the lis
        conv_starting_index = 1 if ignore_first_convolutional_decoder else 0
        for layer_index in reversed(range(conv_starting_index, self._n_convolution_layers)):
            decoder_conv_transpose_block = self._build_decoder_deconv_block(layer_index)
            decoder_conv_transpose_blocks.extend(decoder_conv_transpose_block)

        return decoder_conv_transpose_blocks

    def _build_decoder_deconv_block(self, layer_index: int):
        """
        Build an upsampling and deconvolutional layer of the decoder.

        :param layer_index: An integer. It's the index of the convolutional layer list of the encoder.
        :return: created UpSampling layer, Conv1DTranspose layer and BatchNormalization layer
        """
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





