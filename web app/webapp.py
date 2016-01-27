from predict import predict
from db import insert_table
from pymongo import MongoClient
import cPickle as pickle 
from flask import Flask, request, session, render_template
import socket
import requests
import pandas as pd
from colorama import Back

app = Flask(__name__)

def register():
    reg_url = 'http://10.3.34.86:5000/register'
    my_ip = socket.gethostbyname(socket.gethostname())
    my_port = 5000
    requests.post(reg_url, data={'ip': my_ip, 'port': my_port})


@app.route('/hello')
def hello():
    return 'Hello World!'


@app.route('/score', methods = ['POST'])
def score():
    test = [request.get_json()]
    X_xgb, xgb_preds = predict(test, xgb_model)
    insert_table(X_xgb, xgb_preds, table)
    print str(xgb_preds)
    return ''


@app.route('/dashboard')
def dashboard():
    res = table.find()
    df = []
    for x in res:
        df.append(x)
    df = pd.DataFrame(df)

    df['risk'] = df['prediction'].map(map_risk)
    low = df.loc[df['risk'] == 'low']
    medium = df.loc[df['risk'] == 'medium']
    high = df.loc[df['risk'] == 'high']

    low_total = len(low)
    medium_total = len(medium)
    high_total = len(high)

    return render_template('index.html',
                           low_risk=low_total, medium_risk=medium_total, high_risk=high_total,
                           tables=[high.to_html(classes='high'),
                                   medium.to_html(classes='med'),
                                   low.to_html(classes='low')],
                           titles=['na', 'High Risk Transactions', 
                                   'Medium Risk Transactions', 
                                   'Low Risk Transactions'],
                           links=['na', 'high', 'med', 'low']
                           )


def map_risk(pred):
    if pred < 0.33:
        return 'low'
    elif pred >= 0.33 and pred < 0.66:
        return 'medium'
    else:
        return 'high'



if __name__ == '__main__':
    with open('data/xgb_model.pkl') as f:
        xgb_model = pickle.load(f)
    db_client = MongoClient()
    db = db_client['fraud']
    table = db['predictions']
    register()
    app.secret_key = 'A0Xr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(host='0.0.0.0', port=5000, debug=True)