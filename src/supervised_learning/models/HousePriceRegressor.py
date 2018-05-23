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

    def __init__(self, df_train, df_test):
        super().__init__(df_train, df_test)

        self.train_ID = 0
        self.test_ID = 0

        self.X_train = 0
        self.y_train = 0

        self.X_test = 0
        self.y_test = 0

        self.prepare_data()

    def prepare_data(self):
        self.train_ID = self.df_train["Id"]
        self.test_ID = self.df_test["Id"]

        self.df_train = HousePriceRegressor.clean_train_data(self.df_train)

        n_train = self.df_train.shape[0]
        self.y_train = np.log1p(self.df_train["SalePrice"]).values
        self.df_train.drop(["SalePrice"], axis=1, inplace=True)

        all_data = pd.concat((self.df_train, self.df_test)).reset_index(drop=True)
        all_data = HousePriceRegressor.engineer_data(all_data)

        self.df_test = all_data[n_train:]
        self.df_train = all_data[:n_train]

        self.X_train = self.df_train.values
        self.X_test = self.df_test.values

    def predict(self, model):
        start_time = time()
        model.fit(self.X_train, self.y_train)

        # train_pred = model.predict(self.X_train)
        # print(HousePriceRegressor.rmsle(self.y_train, train_pred))

        self.y_test = np.expm1(model.predict(self.X_test))
        self.prediction_to_csv(model.__class__.__name__)

        print("Time: {:.4f}s \n".format(time() - start_time))
        return self.y_test

    def prediction_to_csv(self, filename):
        sub = pd.DataFrame()
        sub["Id"] = self.test_ID
        sub["SalePrice"] = self.y_test
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        sub.to_csv(SUBMISSION_PATH + filename + "_" + now + FORMAT_CSV, index=False)

    def print_rmsle_cv(self, model):
        start_time = time()
        rms = self.rmsle_cv(model)
        print("Score: {:.4f} ({:.4f}) in {:.4f}s \n".format(rms.mean(), rms.std(), time() - start_time))

    def rmsle_cv(self, model):
        kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits()
        return np.sqrt(-cross_val_score(model, self.X_train, self.y_train, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))

    @staticmethod
    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    @staticmethod
    def clean_train_data(data):
        data = HousePriceRegressor.remove_outliers(data)
        return data

    @staticmethod
    def engineer_data(data):
        # print("The train data size after: {} ".format(data.shape))
        data = HousePriceRegressor.drop_useless_columns(data)
        # print("The train data size after: {} ".format(data.shape))
        data = HousePriceRegressor.fill_all_na_values(data)
        # print("The train data size after: {} ".format(data.shape))
        data = HousePriceRegressor.transfor_numerical_to_categorical(data)
        # print("The train data size after: {} ".format(data.shape))
        data = HousePriceRegressor.label_data(data)
        # print("The train data size after: {} ".format(data.shape))
        data = HousePriceRegressor.create_new_features(data)
        # print("The train data size after: {} ".format(data.shape))
        data = HousePriceRegressor.process_skewed_features(data)
        # print("The train data size after: {} ".format(data.shape))
        data = pd.get_dummies(data)
        # print("The train data size after: {} ".format(data.shape))

        return data

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
        cols = ("Id", "Utilities", "PoolQC", "Fence", "MiscFeature", "Alley", "FireplaceQu")
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
    def fill_na_values_list_mode(data, column_name_list):
        for col in (column_name_list):
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
        return data

    @staticmethod
    def fill_na_values_list(data, column_name_list, set_value):
        for col in (column_name_list):
            if col in data.columns:
                data[col] = data[col].fillna(set_value)
        return data

    @staticmethod
    def fill_all_na_values(data):
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
        cols_to_mode = ("MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType")

        data = HousePriceRegressor.fill_na_values_list(data, cols_to_none, "None")
        data = HousePriceRegressor.fill_na_values_list(data, cols_to_zero, 0)
        data = HousePriceRegressor.fill_na_values_list_mode(data, cols_to_mode)

        data = HousePriceRegressor.fill_na_values(data, "Functional", "Typ")

        data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

        # all_data_na = (data.isnull().sum() / len(data)) * 100
        # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        # missing_data = pd.DataFrame({"Missing Ratio": all_data_na})
        # print(missing_data.head(20))

        return data

    @staticmethod
    def create_new_features(data):
        data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
        return data

    @staticmethod
    def transfor_numerical_to_categorical(data):
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
    def label_data(data):
        cols = ("FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
                "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1",
                "BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope",
                "LotShape", "PavedDrive", "Street", "Alley", "CentralAir", "MSSubClass", "OverallCond",
                "YrSold", "MoSold")
        for col in cols:
            if col in data.columns:
                lbl = LabelEncoder()
                lbl.fit(list(data[col].values))
                data[col] = lbl.transform(list(data[col].values))
        return data
