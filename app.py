import warnings

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.supervised_learning.models.HousePriceRegressor import HousePriceRegressor
from src.utils.conf import *


def ignore_warn(*args, **kwargs):
    pass


warnings.filterwarnings("ignore")
warnings.warn = ignore_warn


class LoanRecommender(Flask):
    def __init__(self, *args, **kwargs):
        super(LoanRecommender, self).__init__(*args, **kwargs)

        self.data = {'personal_income': 2000, 'personal_outcome': 1000, 'desired_payment': 500, 'age': 30}

        self.df_train = pd.read_csv(TRAIN_PATH)
        self.df_test = pd.read_csv(TEST_PATH)

        self.basement_data = pd.json.loads(self.df_train.head(1).to_json(orient='records'))[0]
        self.basement_data['GarageYrBlt'] = int(self.basement_data['GarageYrBlt'])

        self.house_regr = HousePriceRegressor(self.df_train, self.df_test)
        lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0006, random_state=1, max_iter=50000))
        self.house_regr.fit(_model=lasso)

        self.sale_price = 0


app = LoanRecommender(__name__)


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/index', methods=['POST', 'GET'])
def index():
    return render_template('index.html', basement_data=app.basement_data, data=app.data)


@app.route('/index_submited', methods=['POST'])
def index_submited():
    content = request.get_json()
    #print(content)
    available_income = float(content['PersonalIncome']) - float(content['PersonalOutcome'])
    desired_payment = float(content['DesiredPayment'])
    age = float(content['Age'])

    max_age = 75
    max_duration = 30

    loan_duration = min(max_age - age, max_duration) * 12

    loan_price = float(content['SalePriceHidden']) * 0.8

    if 'StableIncomeCheckBox' in content:
        rate = 0.05 / 12
        loan_type = 'Conventional with fix rate'
    else:
        rate = 0.035 / 12
        loan_type = 'LIBOR with variable rate'

    if 'SpouceCheckBox' in content:
        available_income = available_income + float(content['SpouceIncome']) - float(content['SpouceOutcome'])

    if content['OtherIncome']:
        available_income = available_income + float(content['OtherIncome'])

    # Monthly price based on duration
    rate_comp1 = 1 / rate
    temp_rate_comb = pow((1 + rate), loan_duration)
    rate_comp2 = 1 / (rate * temp_rate_comb)
    price_duration_based = round(loan_price / (rate_comp1 - rate_comp2), 2)

    # Duration based on desired payment
    price_dp_based = desired_payment
    temp_dur_comp1 = price_dp_based / (price_dp_based - (loan_price * rate))
    temp_dur_comp2 = 1 + rate
    loan_duration_dp = np.log10(temp_dur_comp1) / np.log10(temp_dur_comp2)
    if np.isnan(loan_duration_dp) or loan_duration_dp > (max_duration*12):
        loan_duration_dp = 0
    loan_duration_dp = int(loan_duration_dp)

    # Duration based on desired payment
    price_inc_based = available_income
    temp_dur_comp1 = price_inc_based / (price_inc_based - (loan_price * rate))
    temp_dur_comp2 = 1 + rate
    loan_duration_inc = np.log10(temp_dur_comp1) / np.log10(temp_dur_comp2)
    if np.isnan(loan_duration_inc) or loan_duration_inc > (max_duration*12):
        loan_duration_inc = 0
    loan_duration_inc = int(loan_duration_inc)

    return pd.io.json.dumps(
        {'final_payment': price_duration_based, 'loan_type': loan_type, 'loan_duration': loan_duration,
         'final_payment_dp': price_dp_based, 'loan_duration_dp': loan_duration_dp, 'final_payment_inc': price_inc_based,
         'loan_duration_inc': loan_duration_inc})


@app.route('/index_sale_price', methods=['POST'])
def index_sale_price():
    content = request.get_json()
    #print(content)
    df = pd.io.json.json_normalize(content)
    columns = list(df)
    columns.remove('Neighborhood')
    df[columns] = df[columns].apply(pd.to_numeric)
    # print(df)

    df_random_test = app.df_test.head(1)
    for col in df_random_test.columns:
        if col in df.columns:
            df_random_test[col] = df[col]

    sale_price = app.house_regr.predict(_df_test=df_random_test)
    print(sale_price)

    return pd.io.json.dumps({'sale_price': sale_price})


if __name__ == '__main__':
    app.run()
