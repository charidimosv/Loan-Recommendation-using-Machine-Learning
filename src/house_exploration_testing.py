import warnings

from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns

color = sns.color_palette()
sns.set_style("darkgrid")
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

pd.set_option("display.float_format", lambda x: "{:.3f}".format(x))

from src.utils.conf import *
import pandas as pd
from sklearn.model_selection import cross_val_score
from scipy.stats import skew

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def save_figure(filename):
    plt.savefig(PICTURE_PATH + filename + FORMAT_PNG)


def save_scatter_plot_between_old(df, x_column, y_column):
    data = pd.concat([df[x_column], df[y_column]], axis=1)
    data.plot.scatter(x=x_column, y=y_column, ylim=(0, 800000))
    save_figure(y_column + "_" + x_column)


def save_scatter_plot_between(df, x_column, y_column):
    fig, ax = plt.subplots()
    ax.scatter(x=df[x_column], y=df[y_column])
    plt.xlabel(x_column, fontsize=13)
    plt.ylabel(y_column, fontsize=13)
    save_figure(y_column + "_" + x_column)


def fill_na_values(data, column_name, set_value):
    data[column_name] = data[column_name].fillna(set_value)
    return data


def fill_na_values_mode(data, column_name):
    return fill_na_values(data, column_name, data[column_name].mode()[0])


def fill_na_values_list(data, column_name_list, set_value):
    for col in (column_name_list):
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
    # print("data : " + str(data.shape))
    if "SalePrice" in data.columns:
        data.drop(["SalePrice"], axis=1, inplace=True)
    if "Id" in data.columns:
        data.drop(["Id"], axis=1, inplace=True)
    if "Utilities" in data.columns:
        data.drop(["Utilities"], axis=1, inplace=True)
    # print("data : " + str(data.shape))
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


def apply_log_fun(data):
    # sns.distplot(data["SalePrice"], fit=norm);
    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(data["SalePrice"])
    # print("\n mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))
    # # Now plot the distribution
    # plt.legend(["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )".format(mu, sigma)],
    #            loc="best")
    # plt.ylabel("Frequency")
    # plt.title("SalePrice distribution")
    # # Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(data["SalePrice"], plot=plt)
    # plt.show()

    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    # print(data.SalePrice.values)
    data["SalePrice"] = np.log1p(data["SalePrice"])
    # print(data.SalePrice.values)

    # # Check the new distribution
    # sns.distplot(data["SalePrice"], fit=norm);
    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(data["SalePrice"])
    # print("\n mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))
    # # Now plot the distribution
    # plt.legend(["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )".format(mu, sigma)],
    #            loc="best")
    # plt.ylabel("Frequency")
    # plt.title("SalePrice distribution")
    # # Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(data["SalePrice"], plot=plt)
    # plt.show()
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
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))
    return data


def clean_train_data(data):
    # print("The train data size after: {} ".format(data.shape))
    data = remove_outliers(data)
    # print("The train data size after: {} ".format(data.shape))
    # data = apply_log_fun(data)
    # print("The train data size after: {} ".format(data.shape))

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
    kf = KFold(n_splits=5, shuffle=True, random_state=20170218).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, n_jobs=-1, scoring="neg_mean_squared_error", cv=kf))
    return rmse


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


if __name__ == "__main__":
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    train_ID = df_train["Id"]
    test_ID = df_test["Id"]

    print("The train data size before: {} ".format(df_train.shape))
    df_train = clean_train_data(df_train)

    n_train = df_train.shape[0]
    y_train = np.log1p(df_train["SalePrice"])
    df_train.drop(["SalePrice"], axis=1, inplace=True)

    all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
    df_train = process_data(df_train)
    print("The train data size after: {} \n".format(df_train.shape))

    print("The test data size before: {} ".format(df_test.shape))
    df_test = process_data(df_test)
    print("The test data size after: {} \n".format(df_test.shape))

    print("The ALL data size before: {} ".format(all_data.shape))
    all_data = process_data(all_data)
    print("The ALL data size after: {} \n".format(all_data.shape))

    df_train = all_data[:n_train]
    df_test = all_data[n_train:]

    # X = df_train.values
    # y = y_train
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=31)
    # regressor = RandomForestRegressor(n_estimators=300, random_state=0)
    # regressor.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)
    # score = rmsle_cv(regressor, df_train, y_train)
    # print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
                                       loss="huber", random_state=5)
    # averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)

    score = rmsle_cv(lasso, df_train.values, y_train)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score = rmsle_cv(ENet)
    # print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score = rmsle_cv(KRR)
    # print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score = rmsle_cv(GBoost)
    # print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    #
    # score = rmsle_cv(averaged_models)
    # print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    score = rmsle_cv(stacked_averaged_models, df_train.values, y_train)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # lasso.fit(train.values, y_train)
    # stacked_train_pred = lasso.predict(train.values)
    # stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    # print(rmsle(y_train, stacked_train_pred))

    # sub = pd.DataFrame()
    # sub["Id"] = test_ID
    # sub["SalePrice"] = stacked_pred
    # sub.to_csv(SUBMISSION_PATH + "submission.csv", index=False)
