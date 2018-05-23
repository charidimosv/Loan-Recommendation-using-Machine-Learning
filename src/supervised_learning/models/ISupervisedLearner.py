class ISupervisedLearner:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def prepare_data(self):
        pass

    def predict(self, model):
        pass

    def prediction_to_csv(self, filename):
        pass

    @staticmethod
    def engineer_data(data):
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
    def fill_all_na_values(data):
        pass

    @staticmethod
    def create_new_features(data):
        pass
