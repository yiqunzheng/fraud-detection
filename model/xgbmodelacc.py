import xgboost as xgb 
import pandas as pd 
import numpy as np 
from sklearn.cross_validation import cross_val_score
from load_data import load

def main():
    X, y = load()


    print 'Running Model'
    mdl = xgb.XGBClassifier()
    scores = cross_val_score(mdl, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print np.mean(scores)

if __name__ == '__main__':
    main()
