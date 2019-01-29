# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2

Global_list = []


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format, reuse=False, name='batch_norm', weights=None):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops

    with tf.variable_scope('') as scope:
        #     # get the name of the weight
        #     mean = 'resnet/' + scope.name + 'bn/moving_mean'
        #     variance = 'resnet/' + scope.name + 'bn/moving_variance'
        #     offset = 'resnet/' + scope.name + 'bn/beta'
        #     scale = 'resnet/' + scope.name + 'bn/gamma'
        # out = tf.nn.batch_normalization(x=inputs, mean=weights[mean], variance=weights[variance],
        #                                 offset=weights[offset], scale=weights[scale],
        #                                 variance_epsilon=1e-6)
        out = tf.layers.batch_normalization(inputs, training=True, reuse=reuse, name=name)

    return out


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, weights):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    strides = [1, strides, strides, 1]

    with tf.variable_scope('') as scope:
        # print(scope.name)
        # if scope.name == '':
        #     var_name = 'conv2d/kernel'
        # else:
        var_name = 'resnet/' + scope.name + 'conv2d/kernel'
    out = tf.nn.conv2d(
        input=inputs,
        filter=weights[var_name],
        strides=strides,
        padding=('SAME' if strides == [1, 1, 1, 1] else 'VALID')
    )
    return out


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format, weights, reuse=False):
    """
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block.
    """
    shortcut = inputs
    with tf.variable_scope('block2_a') as scope:
        inputs = batch_norm(inputs, training, data_format, reuse=reuse, name='batch_norm_a', weights=weights)
        inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        with tf.variable_scope('sht') as scope:
            shortcut = projection_shortcut(inputs)
    with tf.variable_scope('block2_a') as scope:
        # print(scope.name)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format, weights=weights)
    with tf.variable_scope('block2_b') as scope:
        inputs = batch_norm(inputs, training, data_format, reuse=reuse, name='batch_norm_b', weights=weights)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format, weights=weights)

    return inputs + shortcut


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format, weights, reuse=False):
    """
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    adapted to the ordering conventions of:
      Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """
    shortcut = inputs
    with tf.variable_scope('bot_a', reuse=reuse) as scope:
        inputs = batch_norm(inputs, training, data_format, weights=weights)
        inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        with tf.variable_scope('sht', reuse=reuse) as scope:
            shortcut = projection_shortcut(inputs)

    with tf.variable_scope('bot_a', reuse=reuse) as scope:
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format, weights=weights)

    with tf.variable_scope('bot_b', reuse=reuse) as scope:
        inputs = batch_norm(inputs, training, data_format, weights=weights)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format, weights=weights)

    with tf.variable_scope('bot_c', reuse=reuse) as scope:
        inputs = batch_norm(inputs, training, data_format, weights=weights)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format, weights=weights)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, weights, reuse):
    """Creates one layer of blocks for the ResNet model.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format, weights=weights)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format, weights, reuse=reuse)

    for j in range(1, blocks):
        with tf.variable_scope(str(j)) as scope:
            # print(scope.name)
            inputs = block_fn(inputs, filters, training, None, 1, data_format, weights, reuse=reuse)

    return tf.identity(inputs, name)


class Model(object):
    """Base class for building the Resnet Model.
    """

    def __init__(self, img_size, resnet_size, bottleneck, num_classes, num_filters,
                 kernel_size,
                 conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_sizes, block_strides,
                 final_size, version=DEFAULT_VERSION, data_format=None):
        """Creates a model for classifying an image.

        Args:
          img_size: tuple = (height, width)
          resnet_size: A single integer for the size of the ResNet model.
          bottleneck: Use regular blocks or bottleneck blocks.
          num_classes: The number of classes used as labels.
          num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          kernel_size: The kernel size to use for convolution.
          conv_stride: stride size for the initial convolutional layer
          first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
          first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
          second_pool_size: Pool size to be used for the second pooling layer.
          second_pool_stride: stride size for the final pooling layer
          block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
          block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
          final_size: The expected size of the model after the second pooling.
          version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
          data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
        """
        self.img_size = img_size
        self.resnet_size = resnet_size

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        data_format = 'channels_last'

        self.resnet_version = version
        if version not in (1, 2):
            raise ValueError(
                "Resnet version should be 1 or 2. See README for citations.")

        self.bottleneck = bottleneck
        if bottleneck:
            self.block_fn = _bottleneck_block_v2
        else:
            self.block_fn = _building_block_v2

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size

    def __call__(self, inputs, weights, training=True, reuse=False):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images. shape = Task, N, 32*32*3
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        inputs = tf.reshape(inputs, [-1, self.img_size, self.img_size, 3])
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format, weights=weights)
        inputs = tf.identity(inputs, 'initial_conv')

        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

        attention = FLAGS.attention
        for i, num_blocks in enumerate(self.block_sizes):
            with tf.variable_scope(str(i)) as scope:
                # print(scope.name)
                num_filters = self.num_filters * (2 ** i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=training,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format,
                    weights=weights, reuse=reuse)

                if attention % 2 == 1:
                    feature_size = inputs.get_shape().as_list()[1]
                    att_feature = tf.layers.average_pooling2d(
                        inputs=inputs, pool_size=(feature_size, feature_size),
                        strides=self.second_pool_stride, padding='VALID',
                        data_format=self.data_format)

                    att_mask = tf.nn.conv2d(
                        input=att_feature,
                        filter=weights['A/w' + str(i + 1)],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
                    if FLAGS.use_bias:
                        att_mask = att_mask + weights['A/b' + str(i + 1)]
                    inputs = tf.multiply(inputs, tf.sigmoid(att_mask))

                attention -= attention % 2
                attention /= 2

        inputs = batch_norm(inputs, training, self.data_format, reuse=reuse, weights=weights, name="bn")
        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=self.second_pool_size,
            strides=self.second_pool_stride, padding='VALID',
            data_format=self.data_format)

        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, self.final_size])

        return inputs


def _get_block_sizes(resnet_size):
    """The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    """
    choices = {
        8: [1, 1, 1],
        10: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
            resnet_size, choices.keys()))
        raise ValueError(err)


class ImagenetModel(Model):
    def __init__(self, img_size, resnet_size, num_classes, data_format='channels_last',
                 version=DEFAULT_VERSION, base_num_filters=64):
        """These are the parameters that work for Imagenet data.

        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
            final_size = FLAGS.base_num_filters * 8
            second_pool_size = 2
        else:
            bottleneck = True
            final_size = FLAGS.base_num_filters * 8
            second_pool_size = 2

        super(ImagenetModel, self).__init__(
            img_size=img_size,
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,  # 垃圾变量，没用到
            num_filters=base_num_filters,
            kernel_size=5,
            conv_stride=2,
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=second_pool_size,
            second_pool_stride=1,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[2, 2, 2, 1],
            final_size=final_size,
            version=version,
            data_format=data_format
        )
