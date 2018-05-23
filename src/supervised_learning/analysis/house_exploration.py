import warnings

import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm

from src.utils.pic_utils import *

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # allData = Data(TRAIN_PATH, TEST_PATH)

    df_train = pd.read_csv(TRAIN_PATH)
    # print(df_train.columns)

    print(df_train['SalePrice'].describe())

    plot = sns.distplot(df_train['SalePrice'])
    save_figure("SalePrice")

    # print("Skewness: %f" % df_train['SalePrice'].skew())
    # print("Kurtosis: %f" % df_train['SalePrice'].kurt())

    save_scatter_plot_between_old(df_train, "GrLivArea", "SalePrice")

    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    save_figure("heatmap")

    # print(corrmat.head)
    # print(corrmat.nlargest(5, 'SalePrice').head)
    # print(corrmat.nlargest(5, 'SalePrice')['SalePrice'].head)

    # saleprice correlation matrix
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    save_figure("top10_heatmap")

    print("Find most important features relative to target")
    corr = df_train.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    print(corr.SalePrice)

    # scatterplot
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    pp = sns.pairplot(df_train[cols], size=2.5)
    save_figure("scatterplot")

    # scatterplot2
    sns.set()
    # cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'YrSold', 'OverallCond', 'KitchenAbvGr']
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
    pp = sns.pairplot(df_train[cols], size=2.5)
    save_figure("scatterplotBoth")

    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    # print(missing_data.head(20))

    # histogram and normal probability plot

    plt.figure()
    sns.distplot(df_train['SalePrice'], fit=norm)
    save_figure("normalProbabilityPlot1")
    # plt.show()
    plt.figure()
    stats.probplot(df_train['SalePrice'], plot=plt)
    # plt.show()
    save_figure("normalProbabilityPlot2")

    # applying log transformation
    salePrice = np.log(df_train['SalePrice'])
    # print(df_train['SalePrice'])
    # print(salePrice)

    plt.figure()
    stats.probplot(salePrice, plot=plt)
    # plt.show()
    save_figure("normalProbabilityPlot3")

    # difference = pd.concat([salePrice, df_train['SalePrice']], axis=1, keys=['After', 'Before'])
    # print(difference.head)

    # print(salePrice.head)
