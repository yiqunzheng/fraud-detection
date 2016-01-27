from pymongo import MongoClient
from predict import predict
import numpy as np



def insert_table(X, preds, table):
    for i in xrange(len(X)):
        row = X.ix[i]
        mongorow = dict(zip(row.index, row.values))
        mongorow['prediction'] = np.float64(preds[i])
        bool_cols = ['prev_payout_empty', 'quantity_sold']
        for col in bool_cols:
            mongorow[col] = int(mongorow[col])
        table.insert(mongorow)


if __name__ == '__main__':
    X, preds = predict('data/test.json')
    insert_table(X, preds)