import hashlib
import re
import collections
import os
import tensorflow as tf


class ImageList:
    def __init__(self, image_dir, max_images, testing_percentage, validation_percentage):
        """Create a list of the images on image_dir in three categories (training,
                   testing and validation)

            Args:
                image_dir: Directory with folder having images
                max_images: Limit how many images will be selected
                validation_percentage: Percentage of images that will be used for validation
                testing_percentage: Percentage of images that will be used for testing
        """
        self.image_dir = image_dir
        self.max_images = max_images
        self.validation_percentage = validation_percentage
        self.testing_percentage = testing_percentage
        self.list = self.create_image_lists()

    def create_image_lists(self):
        if not tf.gfile.Exists(self.image_dir):
            tf.logging.error("Image directory '" + self.image_dir + "' not found.")
            return None
        result = collections.OrderedDict()
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.image_dir))
        # The root directory comes first, so skip it.
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = ['jpg', 'jpeg']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == self.image_dir:
                continue
            tf.logging.info("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(self.image_dir, dir_name, '*.' + extension)
                file_list.extend(tf.gfile.Glob(file_glob))
            if not file_list:
                tf.logging.warning('No files found')
                continue
            if len(file_list) > self.max_images:
                tf.logging.warning(
                    'WARNING: Folder {} have {} images more than the limit of {} images. Some images will '
                    'never be selected.'.format(dir_name, len(file_list), self.max_images))
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
                                    (self.max_images + 1)) *
                                   (100.0 / self.max_images))
                if percentage_hash < self.validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (self.testing_percentage + self.validation_percentage):
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

    def get_image_path(self, label_name, index, category):
        """Returns a path to an image for a label at the given index.

        Args:
            label_name: Label string we want to get an image for.
            index: Int offset of the image we want. This will be moduloed by the
            available number of images for the label, so it can be arbitrarily large.
            category: Name string of set to pull images from - training, testing, or
            validation.

        Returns:
            File system path string to an image that meets the requested parameters.

        """
        if label_name not in self.list:
            tf.logging.fatal('Label does not exist %s.', label_name)
        label_lists = self.list[label_name]
        if category not in label_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = label_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.',
                             label_name, category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(self.image_dir, sub_dir, base_name)
        return full_path
