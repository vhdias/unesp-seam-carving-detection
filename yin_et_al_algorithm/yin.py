import pandas as pd
import os.path
import tensorflow as tf
from yin_et_al_algorithm import descriptor, features
from joblib import Parallel, delayed


class Yin:
    def __init__(self, image_list, force=False, skip=False, save_loop=10, n_jobs=4):
        self.image_list = image_list
        self.force = force
        self.skip = skip
        self.save_loop = save_loop
        self.n_jobs = n_jobs
        self.df = {}
        self.generate_csv()

    def generate_csv(self):
        if self.skip and not self.force:
            for label in self.image_list.list:
                try:
                    df = pd.read_csv('{}_yin.csv'.format(label), index_col=0)
                except FileNotFoundError:
                    df = pd.DataFrame()
                self.df[label] = df.T
            return
        if self.force:
            for label in self.image_list.list:
                try:
                    os.remove('{}_yin.csv'.format(label))
                except FileNotFoundError:
                    pass
        for label in self.image_list.list:
            try:
                df = pd.read_csv('{}_yin.csv'.format(label), index_col=0)
            except FileNotFoundError:
                df = pd.DataFrame()
            for category in self.image_list.list[label]:
                modified = False
                if category == 'dir':
                    continue
                size_of_category = len(self.image_list.list[label][category])
                tf.logging.info("Getting features from {} images in {} set".format(label, category))
                for i in range(0, size_of_category, self.save_loop):
                    tf.logging.info("{} of {} images processed".format(i, size_of_category))
                    batch_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.get_features_batch)(df, label, category, i, j) for j in range(self.save_loop))
                    for image_df in batch_results:
                        if image_df is None:
                            continue
                        modified = True
                        if df.size == 0:
                            df = pd.concat([df, image_df])
                        else:
                            df = pd.merge(df, image_df, how='outer', left_index=True, right_index=True, copy=False)
                    if modified:
                        df.to_csv('{}_yin.csv'.format(label))
            df.to_csv('{}_yin.csv'.format(label))
            self.df[label] = df.T

    def get_features_batch(self, df, label, category, index_base, i):
        path = self.image_list.get_image_path(label, index_base + i, category)
        if os.path.basename(path) in df:
            return
        image_df = self.get_features(path)
        return image_df

    @staticmethod
    def get_features(image_path):
        image_descriptor = descriptor.ImageDescriptor(image_path)
        ryu_lee = features.get_features_Ryu_Lee(image_descriptor)
        ryu_lee_df = pd.DataFrame(ryu_lee)
        half_seam = features.get_features_half_seam(image_descriptor)
        half_seam_df = pd.DataFrame(half_seam)
        return pd.concat([ryu_lee_df, half_seam_df])
