import datetime
from time import time

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.supervised_learning.models.IRegressor import IRegressor
from src.utils.conf import *


class HousePriceRegressor(IRegressor):

    def __init__(self, _df_train=None, _df_test=None, _model=None):
        self.df_train_clean = None

        self.test_ID = None

        self.X_train = None
        self.y_train = None

        self.X_test = None

        self.model = None

        self.label_dict = None

        self.prepare_data(_df_train, _df_test)

        if _model is not None:
            self.fit(_model)

    def prepare_data(self, df_train=None, df_test=None):

        if df_train is None:
            df_train = pd.read_csv(TRAIN_PATH)

        df_train = HousePriceRegressor.clean_train_data(df_train)

        self.y_train = np.log1p(df_train["SalePrice"]).values
        df_train.drop(["SalePrice"], axis=1, inplace=True)

        self.df_train_clean = df_train

        if df_test is not None:
            self.test_ID = df_test["Id"]
            n_train = df_train.shape[0]

            all_data = pd.concat((self.df_train_clean, df_test)).reset_index(drop=True)
            all_data, self.label_dict = HousePriceRegressor.engineer_data(all_data, all_data)

            df_train = all_data[:n_train]
            df_test = all_data[n_train:]
            self.X_test = df_test.values
        else:
            df_train, self.label_dict = HousePriceRegressor.engineer_data(df_train, df_train)

        self.X_train = df_train.values

    def prepare_test_data(self, df_test):
        self.test_ID = df_test["Id"]

        all_data = pd.concat((self.df_train_clean, df_test)).reset_index(drop=True)
        df_test, _ = HousePriceRegressor.engineer_data(df_test, all_data, self.label_dict)

        self.X_test = df_test.values

    def fit(self, _model):
        start_time = time()
        self.model = _model
        self.model.fit(self.X_train, self.y_train)
        print("Time to Fit ({}): {:.4f}s \n".format(self.model.__class__.__name__, time() - start_time))

    def predict(self, _df_test=None):
        if _df_test is not None:
            self.prepare_test_data(_df_test)

        model_name = self.model.__class__.__name__

        start_time = time()

        y_test = np.expm1(self.model.predict(self.X_test))
        y_test = ['%.2f' % elem for elem in y_test]

        self.prediction_to_csv(model_name, self.test_ID, y_test)
        print("Time to Predict ({}): {:.4f}s \n".format(model_name, time() - start_time))

        return y_test

    @staticmethod
    def prediction_to_csv(filename, test_ID, y_test):
        sub = pd.DataFrame()
        sub["Id"] = test_ID
        sub["SalePrice"] = y_test
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        sub.to_csv(SUBMISSION_PATH + filename + "_" + now + FORMAT_CSV, index=False, sep=',')

    def print_rmsle_cv(self, model=None, k_folds=5):
        start_time = time()
        if model is None:
            model = self.model

        rms = self.rmsle_cv(model, k_folds)
        print("Score ({} {:d}-folds) is {:.4f} ({:.4f}) in {:.4f}s \n".format(model.__class__.__name__, k_folds, rms.mean(), rms.std(), time() - start_time))

    def rmsle_cv(self, model=None, k_folds=5):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42).get_n_splits()
        return np.sqrt(-cross_val_score(model, self.X_train, self.y_train, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))

    @staticmethod
    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    @staticmethod
    def clean_train_data(data):
        data = HousePriceRegressor.remove_outliers(data)
        return data

    @staticmethod
    def engineer_data(data_change, data_check, label_dict=None):
        # print("The train data size after: {} ".format(data.shape))
        data_change = HousePriceRegressor.drop_useless_columns(data_change)
        # print("The train data size after: {} ".format(data.shape))
        data_change = HousePriceRegressor.fill_all_na_values(data_change, data_check)
        # print("The train data size after: {} ".format(data.shape))
        data_change = HousePriceRegressor.transform_numerical_to_categorical(data_change)
        # print("The train data size after: {} ".format(data.shape))
        data_change = HousePriceRegressor.create_new_features(data_change)
        # print("The train data size after: {} ".format(data.shape))
        data_change = HousePriceRegressor.process_skewed_features(data_change)
        # print("The train data size after: {} ".format(data.shape))
        if label_dict is None:
            label_dict = HousePriceRegressor.create_label_dict(data_change)
        data_change = HousePriceRegressor.label_transform_data(data_change, label_dict)
        # data = pd.get_dummies(data)
        # print("The train data size after: {} ".format(data.shape))

        return data_change, label_dict

    @staticmethod
    def remove_outliers(data):
        # fig, ax = plt.subplots()
        # ax.scatter(x=data["GrLivArea"], y=data["SalePrice"])
        # plt.ylabel("SalePrice", fontsize=13)
        # plt.xlabel("GrLivArea", fontsize=13)
        # plt.show()

        # Deleting outliers
        data = data.drop(data[(data["GrLivArea"] > 4000) & (data["SalePrice"] < 300000)].index)

        # # Check the graphic again
        # fig, ax = plt.subplots()
        # ax.scatter(data["GrLivArea"], data["SalePrice"])
        # plt.ylabel("SalePrice", fontsize=13)
        # plt.xlabel("GrLivArea", fontsize=13)
        # plt.show()
        return data

    @staticmethod
    def drop_useless_columns(data):
        cols = ("Id", "Utilities")  # , "PoolQC", "Fence", "MiscFeature", "Alley", "FireplaceQu")
        for col in cols:
            if col in data.columns:
                data.drop([col], axis=1, inplace=True)
        return data

    @staticmethod
    def fill_na_values(data, column_name, set_value):
        if column_name in data.columns:
            data[column_name] = data[column_name].fillna(set_value)
        return data

    @staticmethod
    def fill_na_values_list_mode(data_change, data_check):
        for col in data_change.columns:
            data_change[col] = data_change[col].fillna(data_check[col].mode()[0])
        return data_change

    @staticmethod
    def fill_na_values_list(data, column_name_list, set_value):
        for col in (column_name_list):
            if col in data.columns:
                data[col] = data[col].fillna(set_value)
        return data

    @staticmethod
    def fill_all_na_values(data_change, data_check):
        # all_data_na = (data.isnull().sum() / len(data)) * 100
        # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        # missing_data = pd.DataFrame({"Missing Ratio": all_data_na})
        # print(missing_data.head(20))

        cols_to_none = ("PoolQC", "MiscFeature", 'Alley', "Fence", 'FireplaceQu', 'MasVnrType', "MSSubClass",
                        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")
        cols_to_zero = ("MasVnrArea",
                        "GarageYrBlt", "GarageArea", "GarageCars",
                        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath")

        data_change["LotFrontage"] = data_change.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

        data_change = HousePriceRegressor.fill_na_values(data_change, "Functional", "Typ")
        data_change = HousePriceRegressor.fill_na_values_list(data_change, cols_to_none, "None")
        data_change = HousePriceRegressor.fill_na_values_list(data_change, cols_to_zero, 0)
        data_change = HousePriceRegressor.fill_na_values_list_mode(data_change, data_check)

        # all_data_na = (data.isnull().sum() / len(data)) * 100
        # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        # missing_data = pd.DataFrame({"Missing Ratio": all_data_na})
        # print(missing_data.head(20))

        return data_change

    @staticmethod
    def create_new_features(data):
        data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
        return data

    @staticmethod
    def transform_numerical_to_categorical(data):
        data["MSSubClass"] = data["MSSubClass"].apply(str)
        data["OverallCond"] = data["OverallCond"].astype(str)
        data["YrSold"] = data["YrSold"].astype(str)
        data["MoSold"] = data["MoSold"].astype(str)
        return data

    @staticmethod
    def process_skewed_features(data):
        numeric_feats = data.dtypes[data.dtypes != "object"].index

        # Check the skew of all numerical features
        skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        # print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({"Skew": skewed_feats})
        # print(skewness.head(10))
        skewness = skewness[abs(skewness) > 0.75]
        # print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            data[feat] = boxcox1p(data[feat], lam)
        return data

    @staticmethod
    def create_label_dict(data):
        label_dict = {}

        obj_cols = list(data.dtypes[data.dtypes == "object"].index)
        for col in obj_cols:
            if col in data.columns:
                lbl = LabelEncoder()
                lbl.fit(list(data[col].values))
                label_dict[col] = lbl

        return label_dict

    @staticmethod
    def label_transform_data(data, label_dict):
        cols = list(data.dtypes[data.dtypes == "object"].index)
        for col in cols:
            if col in data.columns:
                lbl = label_dict[col]
                data[col] = lbl.transform(list(data[col].values))
        return data
