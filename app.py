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
    print(request.is_json)
    content = request.get_json()
    print(content)
    print(content['PersonalIncome'])
    result = float(content['PersonalIncome']) - float(content['PersonalOutcome'])
    df = pd.io.json.json_normalize(content)
    print(df)
    return pd.io.json.dumps({'final_payment': result, 'loan_type': 'Euribor 3M Σταθερό'})


@app.route('/index_sale_price', methods=['POST'])
def index_sale_price():
    print(request.is_json)
    content = request.get_json()
    print(content)
    df = pd.io.json.json_normalize(content)
    print(df)
    #house_regr.predict(_df_test=df)
    return pd.io.json.dumps({'sale_price': '1000'})


if __name__ == '__main__':
    data = {}
    data['personal_income'] = 100
    data['personal_outcome'] = 50
    data['loan_payment'] = 30

    df_train = pd.read_csv(TRAIN_PATH)
    temp = df_train.head(1)
    json_data = df_train.head(1).to_json(orient='records')
    basement_data = pd.json.loads(json_data)
    basement_data = basement_data[0]

    r_forest = RandomForestRegressor(n_estimators=300, random_state=0)
    house_regr = HousePriceRegressor( _df_train=df_train, _model=r_forest)
    app.run()
