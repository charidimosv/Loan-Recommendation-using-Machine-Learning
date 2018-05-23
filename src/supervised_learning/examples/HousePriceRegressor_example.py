import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
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

    # r_forest = RandomForestRegressor(n_estimators=300, random_state=0)
    # house_regr.print_rmsle_cv(r_forest)

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features="sqrt",
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss="huber", random_state=5)
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)

    house_regr.print_rmsle_cv(lasso)
    # house_regr.print_rmsle_cv(ENet)
    # house_regr.print_rmsle_cv(KRR)
    # house_regr.print_rmsle_cv(GBoost)
    # house_regr.print_rmsle_cv(stacked_averaged_models)
    test_pred = house_regr.predict(stacked_averaged_models)
