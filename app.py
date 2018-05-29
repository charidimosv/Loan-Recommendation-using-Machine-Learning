import warnings

import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.supervised_learning.models.HousePriceRegressor import HousePriceRegressor
from src.utils.conf import *

app = Flask(__name__)


def ignore_warn(*args, **kwargs):
    pass


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/index', methods=['POST', 'GET'])
def index():
    # your code
    return render_template('index.html', basement_data=basement_data, data=data)


@app.route('/index_submited', methods=['POST'])
def index_submited():
    content = request.get_json()
    print(content)
    available_income = float(content['PersonalIncome']) - float(content['PersonalOutcome'])
    final_payment = float(content['LoanPayment'])
    loan_duration = float(content['LoanDuration'])

    if 'SpouceCheckBox' in content:
        available_income = available_income + float(content['SpouceIncome']) - float(content['SpouceOutcome'])

    if content['OtherIncome']:
        available_income = available_income + float(content['OtherIncome'])

    sale_pricee = 1000 #edo vale olo to daneio
    sale_pricee = sale_pricee * 0.8

    years = int(sale_pricee / (available_income * 12) + 1)
    months = years * 12

    final_payment = sale_price / months

    if available_income > final_payment:
        final_payment = (final_payment + available_income) / 2.0

    return pd.io.json.dumps({'final_payment': final_payment, 'loan_type': 'Euribor 3M Σταθερό'})


@app.route('/index_sale_price', methods=['POST'])
def index_sale_price():
    content = request.get_json()
    print(content)
    df = pd.io.json.json_normalize(content)
    columns = list(df)
    columns.remove('Neighborhood')
    df[columns] = df[columns].apply(pd.to_numeric)
    print(df)

    df_random_test = df_test.head(1)
    for col in df_random_test.columns:
        if col in df.columns:
            df_random_test[col] = df[col]

    sale_price = house_regr.predict(_df_test=df_random_test)
    print(sale_price)

    return pd.io.json.dumps({'sale_price': sale_price})


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    warnings.warn = ignore_warn

    data = {}
    data['personal_income'] = 2000
    data['personal_outcome'] = 1000
    data['loan_payment'] = 500
    data['loan_duration'] = 5

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    basement_data = pd.json.loads(df_train.head(1).to_json(orient='records'))[0]
    basement_data['GarageYrBlt'] = int(basement_data['GarageYrBlt'])

    house_regr = HousePriceRegressor(df_train, df_test)
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0006, random_state=1, max_iter=50000))
    house_regr.fit(_model=lasso)

    sale_price = 0

    app.run()
