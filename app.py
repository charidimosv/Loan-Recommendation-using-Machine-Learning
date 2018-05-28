import pandas as pd
import ast
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor

from src.supervised_learning.models.HousePriceRegressor import HousePriceRegressor, TRAIN_PATH

app = Flask(__name__)


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index', methods=['POST','GET'])
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
    #print(df)

    sale_price = house_regr.predict(_df_test=df)
    return pd.io.json.dumps({'sale_price': sale_price})


if __name__ == '__main__':
    data = {}
    data['personal_income'] = 2000
    data['personal_outcome'] = 1000
    data['loan_payment'] = 500
    data['loan_duration'] = 5

    df_train = pd.read_csv(TRAIN_PATH)
    basement_data = pd.json.loads(df_train.head(1).to_json(orient='records'))[0]
    basement_data['GarageYrBlt'] = int(basement_data['GarageYrBlt'])
    r_forest = RandomForestRegressor(n_estimators=300, random_state=0)
    house_regr = HousePriceRegressor(_df_train=df_train, _model=r_forest)
    sale_price = 0

    app.run()
