from ldp_algorithm.ldp import LDP
import time
from joblib import Parallel, delayed
import pandas as pd
import os
import tensorflow as tf
import cv2


class LdpCsv:
    def __init__(self, image_list, force=False, skip=False, save_loop=10, n_jobs=4):
        self.image_list = image_list
        self.force = force
        self.skip = skip
        self.save_loop = save_loop
        self.n_jobs = n_jobs
        self.df = {}
        self.generate_csv()

    def generate_csv(self):
        if self.skip:
            for label in self.image_list.list:
                try:
                    df = pd.read_csv('{}_ldp.csv'.format(label), index_col=0)
                except FileNotFoundError:
                    df = pd.DataFrame()
                self.df[label] = df.T
            return
        if self.force:
            for label in self.image_list.list:
                try:
                    os.rename('{}_ldp.csv'.format(label), '{}_ldp.{}.csv'.format(label, time.time()))
                except FileNotFoundError:
                    pass
        for label in self.image_list.list:
            try:
                df = pd.read_csv('{}_ldp.csv'.format(label), index_col=0)
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
                        df.to_csv('{}_ldp.csv'.format(label))
            df.to_csv('{}_ldp.csv'.format(label))
            self.df[label] = df.T

    def get_features_batch(self, df, label, category, index_base, i):
        path = self.image_list.get_image_path(label, index_base + i, category)
        if os.path.basename(path) in df:
            return
        image_df = self.get_features(path)
        return image_df

    @staticmethod
    def get_features(image_path):
        image_name = os.path.basename(image_path)
        path = os.path.join('.', 'processed', 'ldp', 'order_{}_angle_{}_radius_{}__' + image_name)
        features = LDP(image_path)
        features.calculate_features()
        df = []
        for order in LDP.ORDER_INDEXES:
            for angle in LDP.ANGLE_INDEXES:
                for radius in LDP.RADIUS_INDEXES:
                    # Save images of ldp
                    cv2.imwrite(path.format(order, angle, radius),
                                features.ldp[order, angle, radius])
                    # Generate columns list
                    columns = []
                    for i in range(256):
                        columns.append('order_%d_angle_%d_radius_%d__%d' % (order, angle, radius, i))
                    # Generate DataFrame
                    temp = pd.DataFrame(features.histograms[order, angle, radius]).T
                    temp.columns = columns
                    df.append(temp.T)
        return pd.concat(df).set_axis([image_name], axis='columns', inplace=False)
