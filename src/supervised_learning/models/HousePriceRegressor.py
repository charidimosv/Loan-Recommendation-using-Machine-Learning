import datetime
from time import time

import numpy as np
import pandas as pd
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
        df_test = all_data[self.df_train_clean.shape[0]:]
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
        if model is not None:
            self.model = model

        rms = self.rmsle_cv(self.model, k_folds)
        print(
            "Score ({} {:d}-folds) is {:.4f} ({:.4f}) in {:.4f}s \n".format(self.model.__class__.__name__, k_folds, rms.mean(), rms.std(), time() - start_time))

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
        data_change = HousePriceRegressor.drop_useless_columns(data_change)
        data_change = HousePriceRegressor.fill_all_na_values(data_change, data_check)
        data_change = HousePriceRegressor.transform_numerical_to_categorical(data_change)
        data_change = HousePriceRegressor.transform_keeping_ordinal(data_change)
        data_change = HousePriceRegressor.create_features(data_change)
        data_change = HousePriceRegressor.process_skewed_features(data_change)
        if label_dict is None:
            label_dict = HousePriceRegressor.create_label_dict(data_change)
        data_change = HousePriceRegressor.label_transform_data(data_change, label_dict)
        # data = pd.get_dummies(data)

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
    def create_features(data):
        data = HousePriceRegressor.create_grouped_features(data)
        data = HousePriceRegressor.create_combined_features(data)
        data = HousePriceRegressor.create_polynomial_features(data)
        return data

    @staticmethod
    def create_grouped_features(data):
        data["SimplOverallQual"] = data.OverallQual.replace({1: 1, 2: 1, 3: 1,  # bad
                                                             4: 2, 5: 2, 6: 2,  # average
                                                             7: 3, 8: 3, 9: 3, 10: 3  # good
                                                             })
        data["SimplOverallCond"] = data.OverallCond.replace({1: 1, 2: 1, 3: 1,  # bad
                                                             4: 2, 5: 2, 6: 2,  # average
                                                             7: 3, 8: 3, 9: 3, 10: 3  # good
                                                             })
        data["SimplPoolQC"] = data.PoolQC.replace({1: 1, 2: 1,  # average
                                                   3: 2, 4: 2  # good
                                                   })
        data["SimplGarageCond"] = data.GarageCond.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
        data["SimplGarageQual"] = data.GarageQual.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
        data["SimplFireplaceQu"] = data.FireplaceQu.replace({1: 1,  # bad
                                                             2: 1, 3: 1,  # average
                                                             4: 2, 5: 2  # good
                                                             })
        data["SimplFireplaceQu"] = data.FireplaceQu.replace({1: 1,  # bad
                                                             2: 1, 3: 1,  # average
                                                             4: 2, 5: 2  # good
                                                             })
        data["SimplFunctional"] = data.Functional.replace({1: 1, 2: 1,  # bad
                                                           3: 2, 4: 2,  # major
                                                           5: 3, 6: 3, 7: 3,  # minor
                                                           8: 4  # typical
                                                           })
        data["SimplKitchenQual"] = data.KitchenQual.replace({1: 1,  # bad
                                                             2: 1, 3: 1,  # average
                                                             4: 2, 5: 2  # good
                                                             })
        data["SimplHeatingQC"] = data.HeatingQC.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
        data["SimplBsmtFinType1"] = data.BsmtFinType1.replace({1: 1,  # unfinished
                                                               2: 1, 3: 1,  # rec room
                                                               4: 2, 5: 2, 6: 2  # living quarters
                                                               })
        data["SimplBsmtFinType2"] = data.BsmtFinType2.replace({1: 1,  # unfinished
                                                               2: 1, 3: 1,  # rec room
                                                               4: 2, 5: 2, 6: 2  # living quarters
                                                               })
        data["SimplBsmtCond"] = data.BsmtCond.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
        data["SimplBsmtQual"] = data.BsmtQual.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
        data["SimplExterCond"] = data.ExterCond.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
        data["SimplExterQual"] = data.ExterQual.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
        return data

    @staticmethod
    def create_combined_features(data):
        data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
        # Overall quality of the house
        data["OverallGrade"] = data["OverallQual"] * data["OverallCond"]
        # Overall quality of the garage
        data["GarageGrade"] = data["GarageQual"] * data["GarageCond"]
        # Overall quality of the exterior
        data["ExterGrade"] = data["ExterQual"] * data["ExterCond"]
        # Overall kitchen score
        data["KitchenScore"] = data["KitchenAbvGr"] * data["KitchenQual"]
        # Overall fireplace score
        data["FireplaceScore"] = data["Fireplaces"] * data["FireplaceQu"]
        # Overall garage score
        data["GarageScore"] = data["GarageArea"] * data["GarageQual"]
        # Overall pool score
        data["PoolScore"] = data["PoolArea"] * data["PoolQC"]
        # Simplified overall quality of the house
        data["SimplOverallGrade"] = data["SimplOverallQual"] * data["SimplOverallCond"]
        # Simplified overall quality of the exterior
        data["SimplExterGrade"] = data["SimplExterQual"] * data["SimplExterCond"]
        # Simplified overall pool score
        data["SimplPoolScore"] = data["PoolArea"] * data["SimplPoolQC"]
        # Simplified overall garage score
        data["SimplGarageScore"] = data["GarageArea"] * data["SimplGarageQual"]
        # Simplified overall fireplace score
        data["SimplFireplaceScore"] = data["Fireplaces"] * data["SimplFireplaceQu"]
        # Simplified overall kitchen score
        data["SimplKitchenScore"] = data["KitchenAbvGr"] * data["SimplKitchenQual"]
        # Total number of bathrooms
        data["TotalBath"] = data["BsmtFullBath"] + (0.5 * data["BsmtHalfBath"]) + data["FullBath"] + (0.5 * data["HalfBath"])
        # Total SF for house (incl. basement)
        data["AllSF"] = data["GrLivArea"] + data["TotalBsmtSF"]
        # Total SF for 1st + 2nd floors
        data["AllFlrsSF"] = data["1stFlrSF"] + data["2ndFlrSF"]
        # Total SF for porch
        data["AllPorchSF"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]
        # Has masonry veneer or not
        data["HasMasVnr"] = data.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                                     "Stone": 1, "None": 0})
        # House completed before sale or not
        data["BoughtOffPlan"] = data.SaleCondition.replace({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                            "Family": 0, "Normal": 0, "Partial": 1})

        return data

    @staticmethod
    def create_polynomial_features(data):
        cols_to_poly = (
            "OverallQual", "AllSF", 'AllFlrsSF', "GrLivArea", 'SimplOverallQual', 'ExterQual', "GarageCars", "TotalBath", "KitchenQual", "GarageScore")

        for col in cols_to_poly:
            if col in data.columns:
                data[col + "-s2"] = data[col] ** 2
                data[col + "-s3"] = data[col] ** 3
                data[col + "-Sq"] = np.sqrt(data[col])

        return data

    @staticmethod
    def transform_numerical_to_categorical(data):
        data["MSSubClass"] = data["MSSubClass"].apply(str)
        data["OverallCond"] = data["OverallCond"].astype(str)
        data["YrSold"] = data["YrSold"].astype(str)
        data["MoSold"] = data["MoSold"].astype(str)
        return data

    @staticmethod
    def transform_keeping_ordinal(data):
        data = data.replace({"Alley": {"None": 0, "Grvl": 1, "Pave": 2},
                             "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "BsmtExposure": {"None": 0, "Mn": 1, "Av": 2, "Gd": 3},
                             "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                             "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
                             "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
                             "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
                             "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
                             "PavedDrive": {"N": 0, "P": 1, "Y": 2},
                             "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                             "Street": {"Grvl": 1, "Pave": 2},
                             "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}}
                            )

        return data

    @staticmethod
    def process_skewed_features(data):
        numerical_features = data.dtypes[data.dtypes != "object"].index
        skewness = data[numerical_features].apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        skewed_features = skewness.index
        data[skewed_features] = np.log1p(data[skewed_features])
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
