"""
"""
import itertools

import numpy as np
import tensorflow as tf


def build_rn_model(images, questions, answers):
    """
    hardcoded model for experiment on sort-of-clevr dataset
    """
    initializer = tf.contrib.layers.xavier_initializer()

    # NOTE: put some descriptions into checkpoints
    tf.constant('rn on sort-of-clevr')

    # NOTE: arXiv:1706.01427v1, supplementary material, d
    #       in this task our model used: four convolutional layers with 32, 64,
    #       128, 256 kernels, ReLU non-linearities, and batch normalization.
    tensors = images

    for i, filters in enumerate([32, 64, 128, 256]):
        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1 if i == 0 else 2,
            padding='same' if i == 0 else 'valid',
            activation=tf.nn.relu,
            kernel_initializer=initializer)

        # NOTE: i like instance_norm more
        tensors = tf.contrib.layers.instance_norm(tensors)

    # NOTE: arXiv:1706.01427v1, 4 models, dealing with pixels
    #       after convolving the image, each of the d^2 k-dimensional cells in
    #       the dxd feature maps was tagged with an arbitrary coordinate
    #       indicating its relative spatial position, and was treated as an
    #       object for the RN.
    #
    #       71x71 -> 71x71 -> 35x35 -> 17x17 -> 8x8
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    u, v = np.meshgrid(x, x)

    u = np.reshape(u, [1, 8, 8, 1])
    v = np.reshape(v, [1, 8, 8, 1])

    coordinates = np.concatenate([u, v], axis=-1)
    coordinates = tf.constant(coordinates)

    n = tf.shape(images)[0]

    coordinates = tf.tile(coordinates, [n, 1, 1, 1])

    tensors = tf.concat([tensors, coordinates], axis=-1)

    # NOTE: build object pairs, the depth is 256(final layer) + 2 (coordiante)
    #       -> [N, 64, 258]
    tensors = tf.reshape(tensors, [-1, 64, 256 + 2])

    # NOTE: split so that each input has 64 features
    #       -> list([N, 1, 258], ..., [N, 1, 258])
    tensors_list = tf.split(tensors, 64, axis=1)

    # NOTE: reshape questions so they can be concat with object pairs
    #       -> [N, -1, 11]
    questions_temp = tf.reshape(questions, [-1, 1, 11])

    # NOTE: concat object pairs & questions
    object_pairs = []

    for i, j in itertools.product(range(64), repeat=2):
        # NOTE: skip duplicated pairs
        if i >= j:
            continue

        # NOTE: concate -> [N, 1, 258 + 258 + 11]
        object_pair = tf.concat(
            [tensors_list[i], tensors_list[j], questions_temp], axis=2)

        object_pairs.append(object_pair)

    # NOTE: concat back to [N, (64+1)*64/2, 258+258+11]
    tensors = tf.concat(object_pairs, axis=1)

    # NOTE: arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    #       a four-layer MLP consisting of 2,000 units per layer with ReLU
    #       non-linearities was use for g_theta.
    for i in range(4):
        tensors = tf.layers.dense(
            tensors,
            units=2000,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='g_theta_{}'.format(i))

    # NOTE: arXiv:1706.01427v1, figure 2
    #       elementwise sum
    # TODO: do we have to keep dims?
    tensors = tf.reduce_sum(tensors, axis=1, keepdims=True)

    # NOTE: arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    #       and a four-layer MLP consisting of 2,000, 1,000, 500, 100 units
    #       with ReLU non-linearities used for f_phi.
    for i, units in enumerate([2000, 1000, 500, 100]):
        tensors = tf.layers.dense(
            tensors,
            units=units,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='f_phi_{}'.format(i))

    # NOTE: arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    #       an additional final linear layer produced logits for a softmax
    #       over the possible answers.
    tensors = tf.layers.dense(
        tensors,
        units=18,
        activation=None,
        use_bias=True,
        kernel_initializer=initializer,
        name='final')

    # NOTE: arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    #       the softmax output was optimized with a cross-entropy loss function
    #       using the Adam optimizer with a learning rate of 1e-4 and
    #       mini-batches of size 64.
    results = tf.reshape(tensors, [-1, 18], name='results')

    loss = tf.losses.softmax_cross_entropy(
        answers,
        results,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    step = tf.train.get_or_create_global_step()

    optimizer = tf.train \
        .AdamOptimizer(learning_rate=1e-4) \
        .minimize(loss, global_step=step)

    return {
        'step': step,
        'loss': loss,
        'images': images,
        'questions': questions,
        'answers': answers,
        'results': results,
        'optimizer': optimizer,
    }


def build_mlp_model(images, questions, answers):
    """
    """


def build_model(model_type='rn'):
    """
    arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    we also trained a comparable MLP based model (CNN+MLP model) on the sort-
    of-clevr task, to explore the extent to which a standard model can learn to
    answer relational questions.
    """
    if model_type == 'rn':
        return build_rn_model()
    else:
        return build_mlp_model()

