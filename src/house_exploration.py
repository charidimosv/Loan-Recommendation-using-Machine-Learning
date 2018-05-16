import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import matplotlib.pyplot as plt
import pandas as pd

from src.conf import *

warnings.filterwarnings('ignore')


def saveFigure(filename):
    plt.savefig(OUTPUT_PATH + filename + FORMAT_PNG)


def saveScatterPlotBetweenOld(df, xColumn, yColumn):
    data = pd.concat([df[xColumn], df[yColumn]], axis=1)
    data.plot.scatter(x=xColumn, y=yColumn, ylim=(0, 800000))
    saveFigure(yColumn + "_" + xColumn)


def saveScatterPlotBetween(df, xColumn, yColumn):
    fig, ax = plt.subplots()
    ax.scatter(x=df[xColumn], y=df[yColumn])
    plt.xlabel(xColumn, fontsize=13)
    plt.ylabel(yColumn, fontsize=13)
    saveFigure(yColumn + "_" + xColumn)


if __name__ == '__main__':
    # allData = Data(TRAIN_PATH, TEST_PATH)

    df_train = pd.read_csv(TRAIN_PATH)
    # print(df_train.columns)

    print(df_train['SalePrice'].describe())

    plot = sns.distplot(df_train['SalePrice'])
    saveFigure("SalePrice")

    # print("Skewness: %f" % df_train['SalePrice'].skew())
    # print("Kurtosis: %f" % df_train['SalePrice'].kurt())

    saveScatterPlotBetweenOld(df_train, "GrLivArea", "SalePrice")

    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    saveFigure("heatmap")

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
    saveFigure("top10_heatmap")

    print("Find most important features relative to target")
    corr = df_train.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    print(corr.SalePrice)

    # scatterplot
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    pp = sns.pairplot(df_train[cols], size=2.5)
    saveFigure("scatterplot")

    # scatterplot2
    sns.set()
    # cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'YrSold', 'OverallCond', 'KitchenAbvGr']
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
    pp = sns.pairplot(df_train[cols], size=2.5)
    saveFigure("scatterplotBoth")

    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    # print(missing_data.head(20))

    # histogram and normal probability plot

    plt.figure()
    sns.distplot(df_train['SalePrice'], fit=norm)
    saveFigure("normalProbabilityPlot1")
    # plt.show()
    plt.figure()
    stats.probplot(df_train['SalePrice'], plot=plt)
    # plt.show()
    saveFigure("normalProbabilityPlot2")

    # applying log transformation
    salePrice = np.log(df_train['SalePrice'])
    # print(df_train['SalePrice'])
    # print(salePrice)

    plt.figure()
    stats.probplot(salePrice, plot=plt)
    # plt.show()
    saveFigure("normalProbabilityPlot3")

    # difference = pd.concat([salePrice, df_train['SalePrice']], axis=1, keys=['After', 'Before'])
    # print(difference.head)

    # print(salePrice.head)
