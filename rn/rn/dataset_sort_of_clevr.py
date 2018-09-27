"""
    question code:
    [ 0] - 0 (relationa;)
    [ 1] - 1 (non-relational)
    [ 2] - ? (query shape)
    [ 3] - ? (query horizontal position)
    [ 4] - ? (query vertical position)
    [ 5] - * (colors[0])
    [ 6] - * (colors[1])
    [ 7] - * (colors[2])
    [ 8] - * (colors[3])
    [ 9] - * (colors[4])
    [10] - * (colors[5])

    answer code:
    [ 0 ~  5] - counts (1 ~ 6)
    [ 6 ~ 11] - colors (0 ~ 5)
    [12 ~ 13] - shapes (0 ~ 1)
    [14 ~ 17] - positions (left, right, top, bottom)
"""
import hashlib
import itertools
import os
import random

import numpy as np
import tensorflow as tf


def decode_qnas(serialized_example):
    """
    decode image and questions in a tfrecord.
    """
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'image_size': tf.FixedLenFeature([], tf.int64),
        'shape_size': tf.FixedLenFeature([], tf.int64),
        'question_size': tf.FixedLenFeature([], tf.int64),
        'answer_size': tf.FixedLenFeature([], tf.int64),
        'num_relational_qnas': tf.FixedLenFeature([], tf.int64),
        'num_non_relational_qnas': tf.FixedLenFeature([], tf.int64),
        'qnas': tf.FixedLenFeature([], tf.string),
    })

    # NOTE: shape of each image is [image_size, image_size, 3]
    image_size = tf.cast(features['image_size'], tf.int32)

    # NOTE: for sort-of-clevr, a question consists of 11 binary flags.
    # NOTE: for sott-of-clevr, an answer consists of 18 binary flags.
    question_size = tf.cast(features['question_size'], tf.int32)
    answer_size = tf.cast(features['answer_size'], tf.int32)
    num_relational_qnas = tf.cast(features['num_relational_qnas'], tf.int32)
    num_non_relational_qnas = \
        tf.cast(features['num_non_relational_qnas'], tf.int32)

    # NOTE: decode bytes back to floats
    image = tf.decode_raw(features['image'], tf.float32)
    qnas = tf.decode_raw(features['qnas'], tf.float32)

    # NOTE: reshape image, all data in tfrecord is flattened
    image = tf.reshape(image, [image_size, image_size, 3])

    # NOTE: reshape questions and answers, all data in tfrecord is flattened
    # NOTE: default shape of questions & answers
    #                               question(11)    answer(18)
    #       relational     (10)
    #       non relational (10)
    qnas = tf.reshape(qnas, [
        num_relational_qnas + num_non_relational_qnas,
        question_size + answer_size])

    return image, qnas


def decode_random_qna(serialized_example):
    """
    decode image and questions in a tfrecord. return the image, a randomly
    selected question and its answer.
    """
    image, qnas = decode_qnas(serialized_example)

    # NOTE: random crop to randomly pick a question and answer pair
    qna = tf.random_crop(qnas, [1, 11 + 18])

    # NOTE: split q&a to question and answer
    question, answer = qna[0, :11], qna[0, 11:]

    return image, question, answer


def build_clevr_batch_iterator(dir_path, batch_size=32):
    """
    read TFRecord batch.
    """
    # NOTE: build path list dataset, the '*' is must for google cloud storage
    path_pattern = os.path.join(dir_path, '*.tfrecord')

    data = tf.data.Dataset.list_files(path_pattern, shuffle=True)

    # NOTE: the path generator never ends
    data = data.repeat()

    # NOTE: read tfrecord
    data = tf.data.TFRecordDataset(data, num_parallel_reads=16)

    # NOTE: decode tfrecord to get image, a question and its answer
    data = data.map(decode_random_qna)

    # NOTE: combine images to batch
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator


def collide_shape(image, x, y, shape, shape_size):
    """
    check if a new shape collide the shapes that are already on the image at
    image[y:y+shape_size, x:x+shape_size].

    arXiv:1706.01427v1, 3.2
    each image has a total of 6 objects, where each object is a randomly chosen
    shape (square or circle).

    I do not want to draw circles  :)
    """
    if shape == 0:
        # NOTE: check square
        for u, v in itertools.product(range(shape_size), repeat=2):
            if image[y+v][x+u] != '0':
                return True
    else:
        # NOTE: check triangle
        for v in range(shape_size):
            for u in range(v):
                if image[y+v][x+u] != '0':
                    return True

    return False


def draw_shape(image, x, y, color, shape, shape_size):
    """
    draw a shape on the image at image[y:y+shape_size, x:x+shape_size]

    arXiv:1706.01427v1, 3.2
    each image has a total of 6 objects, where each object is a randomly chosen
    shape (square or circle).

    I do not want to draw circles  :)
    """
    if shape == 0:
        # NOTE: check square
        for u, v in itertools.product(range(shape_size), repeat=2):
            image[y+v][x+u] = color
    else:
        # NOTE: check triangle
        for v in range(shape_size):
            for u in range(v):
                image[y+v][x+u] = color


def generate_image(image_size, num_colors, num_shapes, shape_size):
    """
    generate an image and draw some shapes on it.
    """
    # NOTE: an image use characters as pixel values
    image = [['0' for _ in range(image_size)] for _ in range(image_size)]

    objects = []

    for color_index in range(1, 1 + num_colors):
        # NOTE: select a shape
        shape_index = random.randrange(num_shapes)

        # NOTE: find an empty position
        while True:
            x = random.randrange(image_size - shape_size)
            y = random.randrange(image_size - shape_size)

            if not collide_shape(image, x, y, shape_index, shape_size):
                break

        # NOTE: draw the colored object
        draw_shape(image, x, y, str(color_index), shape_index, shape_size)

        # NOTE: keep object information for generating questions
        objects.append({
            'x': x + shape_size // 2,
            'y': y + shape_size // 2,
            'c': color_index,
            's': shape_index})

    # NOTE: arXiv:1706.01427v1, 3.2
    #       we used 6 colors (red, blue, green, orange, yellow, gray) to
    #       unambiguously identify each object.
    #
    #       defferent from paper:
    #       red / green / blue / magenta / yellow / cyan
    #
    #       transform the final image to a list of float
    table = {
        '0': [0.0, 0.0, 0.0],
        '1': [1.0, 0.0, 0.0],
        '2': [0.0, 1.0, 0.0],
        '3': [0.0, 0.0, 1.0],
        '4': [1.0, 0.0, 1.0],
        '5': [1.0, 1.0, 0.0],
        '6': [0.0, 1.0, 1.0],
    }

    image = [table[p] for p in itertools.chain.from_iterable(image)]

    image = [p for p in itertools.chain.from_iterable(image)]

    return image, objects


def generate_relational_qnas_closest_to(objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    what is the shape of the object that is closest to the green object?
    """
    q = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [0.0] * 18

    # NOTE: pick an object
    object_index = random.randrange(len(objects))

    q[5 + object_index] = 1.0

    # NOTE: find the closest
    closest_index = object_index
    distance = None

    x, y = objects[object_index]['x'], objects[object_index]['y']

    for idx, obj in enumerate(objects):
        # NOTE: skip self
        if idx == object_index:
            continue

        # NOTE: compare distance
        dx, dy = objects[idx]['x'] - x, objects[idx]['y'] - y

        d = dx * dx + dy * dy

        if distance is None or distance > d:
            distance = d
            closest_index = idx

    # NOTE: 12: square, 13: triangle
    a[12 + objects[closest_index]['s']] = 1.0

    return q, a


def generate_relational_qnas_furthest_from(objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    what is the shape of the object that is furthest from the green object?
    """
    q = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [0.0] * 18

    # NOTE: pick an object
    object_index = random.randrange(len(objects))

    q[5 + object_index] = 1.0

    # NOTE: find the furthest
    furthest_index = object_index
    distance = 0

    x, y = objects[object_index]['x'], objects[object_index]['y']

    for idx, obj in enumerate(objects):
        if idx == object_index:
            continue

        dx, dy = objects[idx]['x'] - x, objects[idx]['y'] - y

        d = dx * dx + dy * dy

        if distance < d:
            distance = d
            furthest_index = idx

    # NOTE: 12: square, 13: triangle
    a[12 + objects[furthest_index]['s']] = 1.0

    return q, a


def generate_relational_qnas_count(objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    how many objects have the shape of the green objects?
    """
    q = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [0.0] * 18

    # NOTE: pick an object
    object_index = random.randrange(len(objects))

    q[5 + object_index] = 1.0

    # NOTE: reference shape
    s = objects[object_index]['s']

    # NOTE: count same shape objects
    count = sum([1 if o['s'] == s else 0 for o in objects])

    # NOTE: [ 0 ~  5] - counts (1 ~ 6), at least one (self)
    a[count - 1]  = 1.0

    return q, a


def generate_relational_qnas(objects, num_qnas):
    """
    generate and return list of relational questions & answers.

    return format:
    [
        [0.0, 1.0, ...],
        [0.0, 1.0, ...],
        ...
    ]
    """
    qnas = []

    codes = set()

    # NOTE: 3 kinds of relational questions and answers
    generators = [
        generate_relational_qnas_closest_to,
        generate_relational_qnas_furthest_from,
        generate_relational_qnas_count,
    ]

    while len(qnas) < num_qnas:
        # NOTE: generate a random q&a
        generator = random.choice(generators)

        q, a = generator(objects)

        # NOTE: skip duplicated q&a
        code = np.array(q).tostring()

        if code in codes:
            continue

        codes.add(code)

        qnas.append(q + a)

    return qnas


def generate_non_relational_qna_query_shape(size, objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    what is the shape of the red object?
    """
    q = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [0.0] * 18

    object_index = random.randrange(len(objects))

    q[5 + object_index] = 1.0

    # NOTE: [12 ~ 13] - shapes (0 ~ 1)
    a[12 + objects[object_index]['s']] = 1.0

    return q, a


def generate_non_relational_qna_query_horizontal_position(size, objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    is the red object on the left or right of the image?
    """
    q = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [0.0] * 18

    object_index = random.randrange(len(objects))

    q[5 + object_index] = 1.0

    # NOTE: a[14] is left and a[15] is right
    if objects[object_index]['x'] > size // 2:
        a[15] = 1.0
    else:
        a[14] = 1.0

    return q, a


def generate_non_relational_qna_query_vertical_position(size, objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    is the red object on the top or bottom of the image?
    """
    q = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [0.0] * 18

    object_index = random.randrange(len(objects))

    q[5 + object_index] = 1.0

    # NOTE: a[16] is top and a[17] is bottom
    if objects[object_index]['y'] > size // 2:
        a[17] = 1.0
    else:
        a[16] = 1.0

    return q, a


def generate_non_relational_qnas(size, objects, num_qnas):
    """
    generate and return list of non-relational questions & answers.

    return format:
    [
        [0.0, 1.0, ...],
        [0.0, 1.0, ...],
        ...
    ]
    """
    qnas = []

    codes = set()

    # NOTE: 3 kinds of non relational questions and answers
    generators = [
        generate_non_relational_qna_query_shape,
        generate_non_relational_qna_query_horizontal_position,
        generate_non_relational_qna_query_vertical_position,
    ]

    while len(qnas) < num_qnas:
        # NOTE: generate a random q&a
        generator = random.choice(generators)

        q, a = generator(size, objects)

        # NOTE: skip duplicated q&a
        code = np.array(q).tostring()

        if code in codes:
            continue

        codes.add(code)

        qnas.append(q + a)

    return qnas


def write_qnas(
        path,
        image,
        image_size,
        shape_size,
        relational_qnas,
        non_relational_qnas):
    """
    path:
        the method write the tfrecord to path
    image:
        list of pixel values in float.
    image_size:
        size of the image (image_size, image_size, 3).
    shape_size:
        size of shape bounding box on the image.
    relational_qnas:
        list of relational questions and their answers. one q&a is composed
        with list of floats. qna[:11] is the question part while the rest of it
        is its answer.
    non_relational_qnas:
        list of non relational questions and their answers. one q&a is composed
        with list of floats. qna[:11] is the question part while the rest of it
        is its answer.
    """
    def floats_feature(q):
        """
        make a feature which consists of one list of floats
        """
        # NOTE: tfrecord can not encode floats. transform it to bytestring.
        # NOTE: shape of q can not be encoded in a feature, it is encoded in
        #       the other feature.
        q = np.array(q, dtype=np.float32)
        q = np.reshape(q, [-1])
        q = q.tostring()

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[q]))

    def int64_feature(v):
        """
        create a feature which consists of a 64-bits integer
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    # NOTE: for sort-of-clevt, a question consists of 11 binary flags.
    # NOTE: for sott-of-clevr, an answer consists of 18 binary flags.

    feature = {
        'image': floats_feature(image),
        'image_size': int64_feature(image_size),
        'shape_size': int64_feature(shape_size),
        'question_size': int64_feature(11),
        'answer_size': int64_feature(18),
        'num_relational_qnas': int64_feature(len(relational_qnas)),
        'num_non_relational_qnas': int64_feature(len(non_relational_qnas)),
        'qnas': floats_feature(relational_qnas + non_relational_qnas),
    }

    with tf.python_io.TFRecordWriter(path) as writer:
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())


def generate_sort_of_clevr():
    """
    build sort-of-clevr dataset.
    """
    FLAGS = tf.app.flags.FLAGS

    num_questions = 0

    while num_questions < FLAGS.num_images:
        # NOTE: generate an image
        image, objects = \
            generate_image(FLAGS.image_size, 6, 2, FLAGS.shape_size)

        code = hashlib.md5(np.array(image).tostring()).hexdigest()

        qna_path = os.path.join(FLAGS.result_dir_path, code + '.tfrecord')

        # NOTE: skip duplicated image
        if tf.gfile.Exists(qna_path):
            continue

        num_questions += 1

        relational_qnas = generate_relational_qnas(
            objects, FLAGS.num_relational_per_image)

        non_relational_qnas = generate_non_relational_qnas(
            FLAGS.image_size, objects, FLAGS.num_non_relational_per_image)

        write_qnas(
            qna_path,
            image,
            FLAGS.image_size,
            FLAGS.shape_size,
            relational_qnas,
            non_relational_qnas)


def explore_sort_of_clevr():
    """
    eyeball check the generated dataset
    """
    def build_answer(a):
        """
        decode the answer and make it human readable.
        """
        answer_index = np.argmax(a)

        if answer_index < 6:
            message = '{}'.format(answer_index + 1)
        elif answer_index < 12:
            message = '{}'.format(answer_index - 6 + 1)
        else:
            answers = ['square', 'triangle', 'left', 'right', 'top', 'bottom']

            message = answers[answer_index - 12]

        return message

    FLAGS = tf.app.flags.FLAGS

    # NOTE: decode one image, question and answer
    batch_iterator = build_clevr_batch_iterator(FLAGS.source_dir_path, 1)

    iter_image, iter_q, iter_a = batch_iterator.get_next()

    with tf.Session() as session:
        session.run(batch_iterator.initializer)

        images, qs, aas = session.run([iter_image, iter_q, iter_a])

    # NOTE: squeeze the batch dimension
    image, q, a = images[0], qs[0], aas[0]

    # NOTE: print the image
    for y in range(image.shape[0]):
        line = []

        for x in image[y]:
            if np.all(x == np.array([0.0, 0.0, 0.0], dtype=np.float32)):
                line.append(' ')
            elif np.all(x == np.array([1.0, 0.0, 0.0], dtype=np.float32)):
                line.append('1')
            elif np.all(x == np.array([0.0, 1.0, 0.0], dtype=np.float32)):
                line.append('2')
            elif np.all(x == np.array([0.0, 0.0, 1.0], dtype=np.float32)):
                line.append('3')
            elif np.all(x == np.array([1.0, 0.0, 1.0], dtype=np.float32)):
                line.append('4')
            elif np.all(x == np.array([1.0, 1.0, 0.0], dtype=np.float32)):
                line.append('5')
            elif np.all(x == np.array([0.0, 1.0, 1.0], dtype=np.float32)):
                line.append('6')

        line.append('|')

        print(''.join(line))

    line = ['-'] * image.shape[1]
    line[image.shape[1] // 2] = '^'

    print(''.join(line))

    # NOTE: decode the question and make it human readable.
    if q[0] == 0.0:
        # NOTE: a non relational question
        if q[2] == 1.0:
            question = 'what is the shape of the {} object?'
        if q[3] == 1.0:
            question = 'is the {} object on the left or right of the image?'
        if q[4] == 1.0:
            question = 'is the {} object on the top or bottom of the image?'

        question = question.format(np.argmax(q[5:]) + 1)
    else:
        # NOTE: a relational question
        if q[2] == 1.0:
            question = 'what is the shape of the object that is closest to'
        if q[3] == 1.0:
            question = 'what is the shape of the object that is furthest from'
        if q[4] == 1.0:
            question = 'how many objects have the shape of'

        question += ' the {} object?'.format(np.argmax(q[5:]) + 1)

    answer = build_answer(a)

    print('{} {}'.format(question, answer))


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.result_dir_path is not None:
        generate_sort_of_clevr()

    if FLAGS.source_dir_path is not None:
        explore_sort_of_clevr()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string(
        'source_dir_path',
        None,
        'path to a dir which contains sort-of-clevr dataset')
    tf.app.flags.DEFINE_string(
        'result_dir_path',
        None,
        'path to a dir for saving generated sort-of-clevr dataset')

    tf.app.flags.DEFINE_integer(
        'image_size',
        75,
        'size of the images to be generated')
    tf.app.flags.DEFINE_integer(
        'shape_size',
        11,
        'size of shape bounding box on the image to be generated')
    tf.app.flags.DEFINE_integer(
        'num_images',
        10000,
        'number of images to be generated')

    tf.app.flags.DEFINE_integer(
        'num_relational_per_image',
        10,
        'number of relational question&answer for each generated image')
    tf.app.flags.DEFINE_integer(
        'num_non_relational_per_image',
        10,
        'number of non-relational question&answer for each generated image')

    tf.app.run()

