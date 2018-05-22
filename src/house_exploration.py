import warnings
from time import time

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.conf import *
from src.models.StackingAveragedModels import StackingAveragedModels


# from src.utils.pic_utils import pic_utils


def ignore_warn(*args, **kwargs):
    pass


warnings.filterwarnings("ignore")
warnings.warn = ignore_warn
pd.set_option("display.float_format", lambda x: "{:.3f}".format(x))


def fill_na_values(data, column_name, set_value):
    if column_name in data.columns:
        data[column_name] = data[column_name].fillna(set_value)
    return data


def fill_na_values_mode(data, column_name):
    return fill_na_values(data, column_name, data[column_name].mode()[0])


def fill_na_values_list(data, column_name_list, set_value):
    for col in (column_name_list):
        if col in data.columns:
            data[col] = data[col].fillna(set_value)
    return data


def fill_all_na_values(data):
    # all_data_na = (data.isnull().sum() / len(data)) * 100
    # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # missing_data = pd.DataFrame({"Missing Ratio": all_data_na})
    # print(missing_data.head(20))

    data = fill_na_values(data, "PoolQC", "None")
    data = fill_na_values(data, "MiscFeature", "None")
    data = fill_na_values(data, "Alley", "None")
    data = fill_na_values(data, "Fence", "None")
    data = fill_na_values(data, "FireplaceQu", "None")
    data = fill_na_values(data, "MasVnrType", "None")
    data = fill_na_values(data, "MSSubClass", "None")
    data = fill_na_values(data, "MasVnrArea", 0)
    data = fill_na_values(data, "Functional", "Typ")

    data = fill_na_values_mode(data, "MSZoning")
    data = fill_na_values_mode(data, "Electrical")
    data = fill_na_values_mode(data, "KitchenQual")
    data = fill_na_values_mode(data, "Exterior1st")
    data = fill_na_values_mode(data, "Exterior2nd")
    data = fill_na_values_mode(data, "SaleType")

    data = fill_na_values_list(data, ("GarageType", "GarageFinish", "GarageQual", "GarageCond"), "None")
    data = fill_na_values_list(data, ("GarageYrBlt", "GarageArea", "GarageCars"), 0)
    data = fill_na_values_list(data, ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"), 0)
    data = fill_na_values_list(data, ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"), "None")

    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    # all_data_na = (data.isnull().sum() / len(data)) * 100
    # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # missing_data = pd.DataFrame({"Missing Ratio": all_data_na})
    # print(missing_data.head(20))

    return data


def drop_useless_columns(data):
    cols = ("Id", "Utilities", "PoolQC", "Fence", "MiscFeature", "Alley", "FireplaceQu")
    for col in cols:
        if col in data.columns:
            data.drop([col], axis=1, inplace=True)
    return data


def create_new_features(data):
    data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    return data


def transfor_numerical_to_categorical(data):
    data["MSSubClass"] = data["MSSubClass"].apply(str)
    data["OverallCond"] = data["OverallCond"].astype(str)
    data["YrSold"] = data["YrSold"].astype(str)
    data["MoSold"] = data["MoSold"].astype(str)
    return data


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


def clean_train_data(data):
    data = remove_outliers(data)
    return data


def process_data(data):
    # print("The train data size after: {} ".format(data.shape))
    data = drop_useless_columns(data)
    # print("The train data size after: {} ".format(data.shape))
    data = fill_all_na_values(data)
    # print("The train data size after: {} ".format(data.shape))
    data = transfor_numerical_to_categorical(data)
    # print("The train data size after: {} ".format(data.shape))
    data = label_data(data)
    # print("The train data size after: {} ".format(data.shape))
    data = create_new_features(data)
    # print("The train data size after: {} ".format(data.shape))
    data = process_skewed_features(data)
    # print("The train data size after: {} ".format(data.shape))
    data = pd.get_dummies(data)
    # print("The train data size after: {} ".format(data.shape))

    return data


def rmsle_cv(model, X_train, y_train):
    t1 = time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits()
    rms = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))
    t2 = time()
    print(str(round(t2 - t1)) + "s")
    return rms


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


if __name__ == "__main__":
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    train_ID = df_train["Id"]
    test_ID = df_test["Id"]

    # print("The train data size before: {} ".format(df_train.shape))
    df_train = clean_train_data(df_train)

    n_train = df_train.shape[0]
    y_train = np.log1p(df_train["SalePrice"]).values
    df_train.drop(["SalePrice"], axis=1, inplace=True)

    all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
    # df_train = process_data(df_train)

    # print("The train data size after: {} \n".format(df_train.shape))

    # print("The test data size before: {} ".format(df_test.shape))
    # df_test = process_data(df_test)
    # print("The test data size after: {} \n".format(df_test.shape))

    # print("The ALL data size before: {} ".format(all_data.shape))
    all_data = process_data(all_data)
    # print("The ALL data size after: {} \n".format(all_data.shape))

    df_train = all_data[:n_train]
    X_train = df_train.values
    df_test = all_data[n_train:]

    regressor = RandomForestRegressor(n_estimators=300, random_state=0)
    score = rmsle_cv(regressor, X_train, y_train)
    print("\nRandomForestRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features="sqrt",
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss="huber", random_state=5)
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)

    score = rmsle_cv(lasso, X_train, y_train)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    # score = rmsle_cv(stacked_averaged_models, X_train, y_train)
    # print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # score = rmsle_cv(ENet, df_train.values, y_train)
    # print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score = rmsle_cv(KRR, df_train.values, y_train)
    # print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score = rmsle_cv(GBoost, df_train.values, y_train)
    # print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    # score = rmsle_cv(stacked_averaged_models)
    # print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # t1 = time()
    # stacked_averaged_models.fit(X_train, y_train)
    # train_pred = stacked_averaged_models.predict(X_train)
    # test_pred = np.expm1(stacked_averaged_models.predict(df_test.values))
    # print(rmsle(y_train, train_pred))
    # t2 = time()
    # print(str(round(t2 - t1)) + "s")

    # lasso.fit(train.values, y_train)
    # stacked_train_pred = lasso.predict(train.values)
    # stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    # print(rmsle(y_train, stacked_train_pred))

    # sub = pd.DataFrame()
    # sub["Id"] = test_ID
    # sub["SalePrice"] = test_pred
    # sub.to_csv(SUBMISSION_PATH + "submission.csv", index=False)
