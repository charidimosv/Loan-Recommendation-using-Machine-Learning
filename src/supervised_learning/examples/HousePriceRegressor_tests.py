import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.supervised_learning.models.HousePriceRegressor import HousePriceRegressor
from src.supervised_learning.models.StackingAveragedModels import StackingAveragedModels
from src.utils.conf import *


def ignore_warn(*args, **kwargs):
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.warn = ignore_warn

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    house_regr = HousePriceRegressor(df_train, df_test)

    KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features="sqrt",
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss="huber", random_state=5)
    # r_forest = RandomForestRegressor(n_estimators=300, random_state=0)
    # lr = LinearRegression()
    # ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

    # Lasso START
    lasso = make_pipeline(RobustScaler(), LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1], max_iter=50000, cv=10))
    lasso.fit(house_regr.X_train, house_regr.y_train)
    alpha = lasso.steps[1][1].alpha_
    print("Best alpha :", alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = make_pipeline(RobustScaler(), LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                            alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                            alpha * 1.4],
                    max_iter=50000, cv=10))
    lasso.fit(house_regr.X_train, house_regr.y_train)
    alpha = lasso.steps[1][1].alpha_
    print("Best alpha :", alpha)
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=alpha, max_iter=50000))
    house_regr.print_rmsle_cv(lasso, 10)
    # Lasso END

    # ElasticNet START
    elasticNet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                                      0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter=50000, cv=10))
    elasticNet.fit(house_regr.X_train, house_regr.y_train)
    alpha = elasticNet.steps[1][1].alpha_
    ratio = elasticNet.steps[1][1].l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)
    print("Try again for more precision with l1_ratio centered around " + str(ratio))
    elasticNet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=[ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                              alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter=50000, cv=10))
    elasticNet.fit(house_regr.X_train, house_regr.y_train)
    if (elasticNet.steps[1][1].l1_ratio_ > 1):
        elasticNet.steps[1][1].l1_ratio_ = 1
    alpha = elasticNet.steps[1][1].alpha_
    ratio = elasticNet.steps[1][1].l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)
    print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
          " and alpha centered around " + str(alpha))
    elasticNet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=ratio,
                              alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9,
                                      alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3,
                                      alpha * 1.35, alpha * 1.4],
                              max_iter=50000, cv=10))
    elasticNet.fit(house_regr.X_train, house_regr.y_train)
    if (elasticNet.steps[1][1].l1_ratio_ > 1):
        elasticNet.steps[1][1].l1_ratio_ = 1
    alpha = elasticNet.steps[1][1].alpha_
    ratio = elasticNet.steps[1][1].l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=alpha, l1_ratio=ratio, random_state=3, max_iter=50000))
    house_regr.print_rmsle_cv(ENet, 10)
    # ElasticNet END

    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)
    house_regr.print_rmsle_cv(stacked_averaged_models)

    # house_regr.print_rmsle_cv(ridge)
    # house_regr.print_rmsle_cv(lasso)
    # house_regr.print_rmsle_cv(lr)
    # house_regr.print_rmsle_cv(stacked_averaged_models)
    # house_regr.print_rmsle_cv(ENet, 5)
    # house_regr.print_rmsle_cv(KRR, 5)
    # house_regr.print_rmsle_cv(GBoost, 5)
    # house_regr.print_rmsle_cv(stacked_averaged_models, 5)

    # house_regr.prepare_test_data(df_test)

    # test_pred = house_regr.predict()
    # test_pred = house_regr.predict(df_test)
