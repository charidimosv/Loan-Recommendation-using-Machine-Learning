import pandas as pd

from src.conf import *


class Data:
    def __init__(self, train_set_file, test_set_file):
        self.train_data = pd.read_csv(train_set_file, index_col=COL_INDEX, sep=COL_SEP, encoding=COL_ENC)
        self.test_data = pd.read_csv(test_set_file, index_col=COL_INDEX, sep=COL_SEP, encoding=COL_ENC)

        self.train_size = int(self.train_data.shape[0] * 2 / 3)
        self.test_size = int(self.train_data.shape[0] - self.train_size)

        cat_set = set(self.train_data[COL_RESULT].values)
        cat_name_to_id = dict(zip(cat_set, range(len(cat_set))))
        self.cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}

        # self.train_data['IntCategory'] = np.asarray([cat_name_to_id[cat] for cat in self.train_data['Category'].values])

        self.n_classes = len(cat_set)
        self.allCategories_int = list(range(self.n_classes))
