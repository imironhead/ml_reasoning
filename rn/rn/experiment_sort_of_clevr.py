"""
"""
import os

import tensorflow as tf

import rn.dataset_sort_of_clevr as dataset
import rn.model_sort_of_clevr as model_rn


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    return dataset.build_clevr_batch_iterator(
        FLAGS.training_data_path, FLAGS.batch_size)


def build_model(dataset_iterator):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: arXiv:1706.01427v1, supplementary material, d
    #       the sort-of-clevr dataset contains 10000 images of 75x75, 200 of
    #       which were withheld for validation.
    #
    #       in this task our model used: four convolutional layers with 32, 64,
    #       128, 256 kernels, ReLU non-linearities, and batch normalization.
    #
    #       I want a shape that can be perfectly convolved, so I tried 71x71

    # NOTE: arXiv:1706.01427v1, supplementary material, d
    #       questions were encoded as binary strings of length 11, where the
    #       first 6 bits identified the color of the object to which the
    #       question refered, as one-hot vector, and the last 5 bits identified
    #       the question type ans subtytpe.

    # NOTE: possible answers (defined in dataset_sort_of_clevr.py)
    #
    #       [ 0 ~  5] - counts (1 ~ 6)
    #       [ 6 ~ 11] - colors (0 ~ 5)
    #       [12 ~ 13] - shapes (0 ~ 1)
    #       [14 ~ 17] - positions (left, right, top, bottom)

    next_image_batch, next_question_batch, next_answer_batch = \
        dataset_iterator.get_next()

    # NOTE: arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    #       we also trained a comparable MLP based model (CNN+MLP model) on the
    #       sort-of-clevr task, to explore the extent to which a standard model
    #       can learn to answer relational questions.
    if FLAGS.type == 'rn':
        model = model_rn.build_rn_model(
            next_image_batch, next_question_batch, next_answer_batch)
    else:
        model = model_rn.build_mpl_model(
            next_image_batch, next_question_batch, next_answer_batch)

    return model


def build_summaries(model):
    """
    """
    summary_loss = tf.summary.scalar('loss', model['loss'])

    return {
        'loss': summary_loss,
    }


def train(model, data_iterator):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    summaries = build_summaries(model)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    saver = tf.train.Saver()

    with tf.Session() as session:
        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        step = session.run(model['step'])

        # NOTE: exclude log which does not happend yet :)
        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        session.run(data_iterator.initializer)

        while step < 1000000:
            fetch = {
                'step': model['step'],
                'optimizer': model['optimizer'],
                'summary_loss': summaries['loss'],
            }

            fetched = session.run(fetch)

            step = fetched['step']

            if 'summary_loss' in fetched:
                reporter.add_summary(fetched['summary_loss'], step)

        reporter.flush()

        saver.save(session, target_ckpt_path, global_step=model['step'])


def test(model, data_iterator):
    """
    """
    FLAGS = tf.app.flags.FLAGS


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    dataset_iterator = build_dataset()
    model = build_model(dataset_iterator)

    train(model, dataset_iterator)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string(
        'training_data_path', None, 'path to the dataset, a pickle file')

    tf.app.flags.DEFINE_string(
        'ckpt_path', None, 'path to the checkpoint')

    tf.app.flags.DEFINE_string(
        'logs_path', None, 'path to the directory for keeping log')

    tf.app.flags.DEFINE_string(
        'type', 'rn', 'rn or mlp for training')

    # NOTE: arXiv:1706.01427v1, supplementary material, d, sort-of-clevr
    #       the softmax output was optimized with a cross-entropy loss function
    #       using the Adam optimizer with a learning rate of 1e-4 and
    #       mini-batches of size 64.
    tf.app.flags.DEFINE_integer(
        'batch_size', 64, 'size of mini-batches')

    tf.app.run()

