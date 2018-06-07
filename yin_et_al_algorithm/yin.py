import pandas as pd
import os.path
from yin_et_al_algorithm import descriptor, features


class Yin:
    def __init__(self, image_list, force=False):
        self.image_list = image_list
        self.force = force
        self.generate_csv()

    def generate_csv(self):
        if not self.force and os.path.exists('untouched.csv') and os.path.exists('seam carved.csv'):
            return
        for label in self.image_list.list:
            for category in self.image_list.list[label]:
                if category == 'dir':
                    continue
                try:
                    df = pd.read_csv('{}.csv'.format(label))
                except FileNotFoundError:
                    df = pd.DataFrame()
                size_of_category = len(self.image_list.list[label][category])
                for i in range(size_of_category):
                    image_descriptor = descriptor.ImageDescriptor(self.image_list.get_image_path(label, i, category))
                    ryu_lee = features.get_features_Ryu_Lee(image_descriptor)
                    ryu_lee_df = pd.DataFrame(ryu_lee)
                    half_seam = features.get_features_half_seam(image_descriptor)
                    half_seam_df = pd.DataFrame(half_seam)
                    image_df = pd.concat([ryu_lee_df, half_seam_df])
                    if df.size == 0:
                        df = pd.concat([df, image_df])
                    else:
                        df = pd.merge(df, image_df, how='outer', left_index=True, right_index=True)
                df.to_csv('{}.csv'.format(category))

