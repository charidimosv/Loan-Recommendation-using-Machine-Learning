class ISupervisedLearner:
    def __init__(self):
        pass

    def prepare_data(self, df_train=None, df_test=None):
        pass

    def prepare_test_data(self, df_test):
        pass

    def fit(self, _model):
        pass

    def predict(self, _df_test=None):
        pass

    @staticmethod
    def prediction_to_csv(filename, test_ID, y_test):
        pass

    @staticmethod
    def engineer_data(data_change, data_check, label_dict=None):
        pass

    @staticmethod
    def clean_train_data(data):
        pass

    @staticmethod
    def remove_outliers(data):
        pass

    @staticmethod
    def drop_useless_columns(data):
        pass

    @staticmethod
    def fill_all_na_values(data_change, data_check):
        pass

    @staticmethod
    def create_new_features(data):
        pass
