import pandas as pd
import numpy as np 


def load():
    numerics = pd.read_csv('data/clean_numeric.csv')
    X = numerics.drop('fraud', axis=1)
    y = numerics['fraud']

    return X, y


def load_desc():
    data = pd.read_csv('data/desc_content.csv')
    X = data['content']
    X = X.fillna('')
    y = data['fraud']

    return X, y