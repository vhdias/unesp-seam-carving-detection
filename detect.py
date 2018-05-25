import argparse
import os.path
import sys

import tensorflow as tf


def ensure_dir_exists(dir_name):
    """ Makes sure the folder exists on disk.
        Args:
            dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def main(_):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.image_dir:
        tf.logging.error('Must set flag --image_dir.')
        return -1

    # Prepare necessary directories that can be used during training
    prepare_file_system()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='./tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='./tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
             How many steps to store intermediate graph. If "0" then will not
             store.\
          """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
