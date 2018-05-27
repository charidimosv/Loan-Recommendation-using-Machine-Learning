import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.supervised_learning.models.AveragingModels import AveragingModels
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

    r_forest = RandomForestRegressor(n_estimators=300, random_state=0)
    house_regr.print_rmsle_cv(r_forest)

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.00042, random_state=1, max_iter=50000))
    elasticNet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0006, l1_ratio=.735, random_state=3, max_iter=50000))
    house_regr.print_rmsle_cv(lasso)
    house_regr.print_rmsle_cv(elasticNet)

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0006, random_state=1, max_iter=50000))
    elasticNet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0006, l1_ratio=1, random_state=3, max_iter=50000))
    house_regr.print_rmsle_cv(lasso)
    house_regr.print_rmsle_cv(elasticNet)

    averaged_models = AveragingModels(models=(elasticNet, lasso))
    house_regr.print_rmsle_cv(averaged_models)

    KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features="sqrt",
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss="huber", random_state=5)
    stacked_averaged_models = StackingAveragedModels(base_models=(elasticNet, GBoost, KRR), meta_model=lasso)
    house_regr.print_rmsle_cv(stacked_averaged_models)

    # house_regr.print_rmsle_cv(stacked_averaged_models)
    # house_regr.print_rmsle_cv(lr)
    # house_regr.print_rmsle_cv(stacked_averaged_models)
    # house_regr.print_rmsle_cv(ENet, 5)
    # house_regr.print_rmsle_cv(KRR, 5)
    # house_regr.print_rmsle_cv(GBoost, 5)
    # house_regr.print_rmsle_cv(stacked_averaged_models, 5)

    # house_regr.prepare_test_data(df_test)

    # test_pred = house_regr.predict()
    test_pred = house_regr.predict(df_test)
