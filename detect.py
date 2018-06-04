import argparse
import hashlib
import numpy
import re
import collections
import os.path
import sys
import tensorflow as tf
import Descriptor

def create_image_lists(image_dir, max_images, validation_percentage, testing_percentage):
    """Create a list of the images on image_dir in three categories (training,
       testing and validation)

    Args:
        image_dir: Directory with folder having images
        max_images: Limit how many images will be selected
        validation_percentage: Percentage of images that will be used for validation
        testing_percentage: Percentage of images that will be used for testing

    """
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) > max_images:
            tf.logging.warning(
                'WARNING: Folder {} have {} images more than the limit of {} images. Some images will '
                'never be selected.'.format(dir_name, len(file_list), max_images))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (max_images + 1)) *
                               (100.0 / max_images))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


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


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index.

    Args:
        image_lists: OrderedDict of training images for each label.
        label_name: Label string we want to get an image for.
        index: Int offset of the image we want. This will be moduloed by the
        available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string of the subfolders containing the training
        images.
        category: Name string of set to pull images from - training, testing, or
        validation.

    Returns:
        File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_features_Ryu_Lee(image: Descriptor.ImageDescriptor):
    # Detecting Trace of Seam Carving for Forensic Analysis; 2014
    # 4 features based on average energy
    # Table 1
    average_column_energy = image.gradient_x.sum() / image.size
    average_row_energy = image.gradient_y.sum() / image.size
    average_energy = image.energy.sum() / image.size
    average_energy_difference = (numpy.abs(image.gradient_x) - numpy.abs(image.gradient_y)).sum() / image.size
    average_energy_features = [average_column_energy, average_energy, average_energy_difference, average_row_energy]

    # 10 features based on the vertical and horizontal seam energy
    # Table 2
    vertical_seam_max = numpy.max(image.vertical_cumulative_energy_transposed[:, image.height - 1])
    vertical_seam_min = numpy.min(image.vertical_cumulative_energy_transposed[:, image.height - 1])
    vertical_seam_mean = numpy.mean(image.vertical_cumulative_energy_transposed[:, image.height - 1])
    vertical_seam_std = numpy.std(image.vertical_cumulative_energy_transposed[:, image.height - 1])
    vertical_seam_diff = vertical_seam_max - vertical_seam_min
    horizontal_seam_max = numpy.max(image.horizontal_cumulative_energy[:, image.width - 1])
    horizontal_seam_min = numpy.min(image.horizontal_cumulative_energy[:, image.width - 1])
    horizontal_seam_mean = numpy.mean(image.horizontal_cumulative_energy[:, image.width - 1])
    horizontal_seam_std = numpy.std(image.horizontal_cumulative_energy[:, image.width - 1])
    horizontal_seam_diff = horizontal_seam_max - horizontal_seam_min
    vertical_horizontal_seam_features = [vertical_seam_max, vertical_seam_min, vertical_seam_mean, vertical_seam_std,
                                         vertical_seam_diff, horizontal_seam_max, horizontal_seam_min,
                                         horizontal_seam_mean, horizontal_seam_std,
                                         horizontal_seam_diff]

    # 4 features based on the noise level
    # Table 3
    noise_mean = image.noise.sum() / image.size  # feature
    noise_less_mean = image.noise - noise_mean
    noise_standart_deviation = noise_less_mean.std()  # feature
    noise_less_mean_divided_std = noise_less_mean / noise_standart_deviation
    noise_skewness = (noise_less_mean_divided_std ** 2).sum() / image.size  # feature
    noise_kurtosis = (noise_less_mean_divided_std ** 3).sum() / image.size  # feature
    noise_level_features = [noise_mean, noise_standart_deviation, noise_kurtosis, noise_skewness]

    return average_energy_features, vertical_horizontal_seam_features, noise_level_features


def get_features_half_seam(image: Descriptor.ImageDescriptor):
    # Detecting seam carving based image resizing using local binary patterns; 2015
    # Ting Yin, Gaobo Yang, Leida Li, Dengyong Zhang, Xingming Sun
    # 6 features based on the vertical and horizontal seam energy
    # Table 1
    vertical_seam_max = numpy.max(image.vertical_cumulative_energy_transposed[:, image.height - 1][0:round(image.width/2)])
    vertical_seam_min = numpy.min(image.vertical_cumulative_energy_transposed[:, image.height - 1][0:round(image.width/2)])
    vertical_seam_mean = numpy.mean(image.vertical_cumulative_energy_transposed[:, image.height - 1][0:round(image.width/2)])
    horizontal_seam_max = numpy.max(image.horizontal_cumulative_energy[:, image.width - 1][0:round(image.height/2)])
    horizontal_seam_min = numpy.min(image.horizontal_cumulative_energy[:, image.width - 1][0:round(image.height/2)])
    horizontal_seam_mean = numpy.mean(image.horizontal_cumulative_energy[:, image.width - 1][0:round(image.height/2)])
    half_seam_features = [vertical_seam_max, vertical_seam_min, vertical_seam_mean, horizontal_seam_max,
                                         horizontal_seam_min, horizontal_seam_mean]

    return half_seam_features


def main(_):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare necessary directories that can be used during training
    prepare_file_system()

    # Create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.max_images_per_class, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.',
        required=True
    )
    required.add_argument(
        '--method',
        type=str,
        default='',
        choices=['YIN', 'YE_SHI', 'CHENG'],
        help='Method for detecting the application of seam carving.',
        required=True
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
    parser.add_argument(
        '--max_images_per_class',
        type=int,
        default=500,
        help='How many images will be selected'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
