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
import argparse
import itertools
import os
import pickle
import random


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

        # NOTE: keep object information to generate questions
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
    table = {
        '0': '000',
        '1': '100', '2': '010', '3': '001',
        '4': '101', '5': '110', '6': '011',
    }

    image = ''.join([table[p] for p in itertools.chain.from_iterable(image)])

    return image, objects


def generate_relational_qnas_closest_to(objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    what is the shape of the object that is closest to the green object?
    """
    q = ['1', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0']
    a = ['0'] * 18

    # NOTE: pick an object
    object_index = random.randrange(len(objects))

    q[5 + object_index] = '1'

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

    # NOTE: [ 6 ~ 11] - colors (0 ~ 5)
    a[6 + closest_index] = '1'

    return {'q': ''.join(q), 'a': ''.join(a)}


def generate_relational_qnas_furthest_from(objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    what is the shape of the object that is furthest from the green object?
    """
    q = ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0']
    a = ['0'] * 18

    # NOTE: pick an object
    object_index = random.randrange(len(objects))

    q[5 + object_index] = '1'

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

    # NOTE: [ 6 ~ 11] - colors (0 ~ 5)
    a[6 + furthest_index] = '1'

    return {'q': ''.join(q), 'a': ''.join(a)}


def generate_relational_qnas_count(objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    how many objects have the shape of the green objects?
    """
    q = ['1', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0']
    a = ['0'] * 18

    # NOTE: pick an object
    object_index = random.randrange(len(objects))

    q[5 + object_index] = '1'

    # NOTE: reference shape
    s = objects[object_index]['s']

    # NOTE: count same shape objects
    count = sum([1 if o['s'] == s else 0 for o in objects])

    # NOTE: [ 0 ~  5] - counts (1 ~ 6), at least one (self)
    a[count - 1]  = '1'

    return {'q': ''.join(q), 'a': ''.join(a)}


def generate_relational_qnas(objects, num_qnas):
    """
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

        qna = generator(objects)

        # NOTE: skip duplicated q&a
        if qna['q'] in codes:
            continue

        qnas.append(qna)

    return qnas


def generate_non_relational_qna_query_shape(size, objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    what is the shape of the red object?
    """
    q = ['0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']
    a = ['0'] * 18

    object_index = random.randrange(len(objects))

    q[5 + object_index] = '1'

    # NOTE: [12 ~ 13] - shapes (0 ~ 1)
    a[12 + objects[object_index]['s']] = '1'

    return {'q': ''.join(q), 'a': ''.join(a)}


def generate_non_relational_qna_query_horizontal_position(size, objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    is the red object on the left or right of the image?
    """
    q = ['0', '1', '0', '1', '0', '0', '0', '0', '0', '0', '0']
    a = ['0'] * 18

    object_index = random.randrange(len(objects))

    q[5 + object_index] = '1'

    # NOTE: a[14] is left and a[15] is right
    if objects[object_index]['x'] > size // 2:
        a[15] = '1'
    else:
        a[14] = '1'

    return {'q': ''.join(q), 'a': ''.join(a)}


def generate_non_relational_qna_query_vertical_position(size, objects):
    """
    arXiv:1706.01427v1, supplementary material, d

    is the red object on the top or bottom of the image?
    """
    q = ['0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0']
    a = ['0'] * 18

    object_index = random.randrange(len(objects))

    q[5 + object_index] = '1'

    # NOTE: a[16] is top and a[17] is bottom
    if objects[object_index]['y'] > size // 2:
        a[17] = '1'
    else:
        a[16] = '1'

    return {'q': ''.join(q), 'a': ''.join(a)}


def generate_non_relational_qnas(size, objects, num_qnas):
    """
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

        qna = generator(size, objects)

        # NOTE: skip duplicated q&a
        if qna['q'] in codes:
            continue

        qnas.append(qna)

    return qnas


def generate_sort_of_clevr(args):
    """
    """
    # NOTE: expect output
    #       [
    #           {
    #               'image': '01010001010....101010',
    #               'relational': [
    #                   {'q': '11010101010', 'a': '101010101010'},
    #                   {'q': '11100010100', 'a': '010101010010'},
    #                   ...
    #               ],
    #               'non_relational': [
    #                   {'q': '01010101010', 'a': '101010101010'},
    #                   {'q': '01100010100', 'a': '010101010010'},
    #                   ...
    #               ],
    #           },
    #           {},
    #           ...
    #       ]
    questions = {}

    while len(questions) < args.num_images:
        # NOTE: generate an image
        image, objects = generate_image(args.image_size, 6, 2, args.shape_size)

        # NOTE: skip duplicated image
        if image in questions:
            continue

        relational_qnas = generate_relational_qnas(
            objects, args.num_relational_per_image)

        non_relational_qnas = generate_non_relational_qnas(
            args.image_size, objects, args.num_non_relational_per_image)

        questions[image] = {
            'image': image,
            'relational': relational_qnas,
            'non_relational': non_relational_qnas,
        }

    # NOTE: build dataset structure
    sort_of_clevr = {
        'image_size': args.image_size,
        'shape_size': args.shape_size,
        'questions': list(questions.values()),
    }

    # NOTE: save
    with open(args.result_path, 'wb') as pkl:
        pickle.dump(sort_of_clevr, pkl)


def explore_sort_of_clevr(args):
    """
    """
    def build_answer(a):
        """
        """
        answer_index = a.find('1')

        if answer_index < 6:
            message = '{}'.format(answer_index + 1)
        elif answer_index < 12:
            message = '{}'.format(answer_index - 6 + 1)
        elif answer_index < 14:
            message = '{}'.format('square' if answer_index == 12 else 'triangle')
        elif answer_index < 16:
            message = '{}'.format('left' if answer_index == 14 else 'right')
        else:
            message = '{}'.format('top' if answer_index == 16 else 'bottom')

        return message

    with open(args.source_path, 'rb') as pkl:
        sort_of_clevr = pickle.load(pkl)

    image_size = sort_of_clevr['image_size']
    shape_size = sort_of_clevr['shape_size']

    print('image size: {}'.format(image_size))
    print('shape size: {}'.format(shape_size))

    # NOTE: explore a random picked q&a
    qnas = random.choice(sort_of_clevr['questions'])

    image = qnas['image']

    r_qna = random.choice(qnas['relational'])
    n_qna = random.choice(qnas['non_relational'])

    # NOTE: print image
    pixel_table = {
        '000': ' ',
        '100': '1', '010': '2', '001': '3',
        '101': '4', '110': '5', '011': '6',
    }

    image = [image[i:i+3] for i in range(0, 3 * (image_size ** 2), 3)]
    image = [pixel_table[pix] for pix in image]

    for i in range(0, image_size ** 2, image_size):
        print(''.join(image[i:i+image_size]))

    print('\n')

    # NOTE: print a non relational question and answer
    assert n_qna['q'][:2] == '01'

    if n_qna['q'][2] == '1':
        question = 'what is the shape of the {} object?'
    if n_qna['q'][3] == '1':
        question = 'is the {} object on the left or right of the image?'
    if n_qna['q'][4] == '1':
        question = 'is the {} object on the top or bottom of the image?'

    question = question.format(n_qna['q'][5:].find('1') + 1)

    answer = build_answer(n_qna['a'])

    print('{} {}'.format(question, answer))

    # NOTE: re-build relational question and answer
    assert r_qna['q'][:2] == '10'

    if r_qna['q'][2] == '1':
        question = 'what is the shape of the object that is closest to'
    if r_qna['q'][3] == '1':
        question = 'what is the shape of the object that is furthest from'
    if r_qna['q'][4] == '1':
        question = 'how many objects have the shape of'

    question += ' the {} object?'.format(r_qna['q'][5:].find('1') + 1)

    answer = build_answer(r_qna['a'])

    print('{} {}'.format(question, answer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build sort-of-clevr')

    # NOTE: arguments for generating sort-of-clevr
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--image_size', type=int, default=75)
    parser.add_argument('--shape_size', type=int, default=11)
    parser.add_argument('--num_images', type=int, default=10_000)
    parser.add_argument('--num_relational_per_image', type=int, default=10)
    parser.add_argument('--num_non_relational_per_image', type=int, default=10)

    # NOTE: arguments for exploring sort-of-clevr
    parser.add_argument('--source_path', type=str)

    args = parser.parse_args()

    if args.result_path is not None:
        generate_sort_of_clevr(args)

    if args.source_path is not None:
        explore_sort_of_clevr(args)

