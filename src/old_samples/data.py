import pandas as pd

from sklearn.preprocessing import label_binarize


class Data:
    def __init__(self, train_set_file, test_set_file):
        self.train_data = pd.read_csv(train_set_file, index_col='Id', sep='\t', encoding='utf-8')
        self.train_data['Content'] = self.train_data['Title'] + self.train_data['Content']
        self.test_data = pd.read_csv(test_set_file, index_col='Id', sep='\t', encoding='utf-8')

        self.train_size = int(self.train_data.shape[0] * 2 / 3)
        self.test_size = int(self.train_data.shape[0] - self.train_size)

        cat_set = set(self.train_data['Category'].values)
        cat_name_to_id = dict(zip(cat_set, range(len(cat_set))))
        self.cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}

        # self.train_data['IntCategory'] = np.asarray([cat_name_to_id[cat] for cat in self.train_data['Category'].values])

        self.n_classes = len(cat_set)
        self.allCategories_int = list(range(self.n_classes))

        self.X_train = self.train_data['Content'].values
        self.y_train = self.train_data['IntCategory'].values
        self.y_train_bin = label_binarize(y=self.y_train, classes=self.allCategories_int)

        self.X_test = self.test_data['Content'].values
