import xgboost as xgb
import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV
from load_data import load, load_desc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


def xgb_model():
    X, y = load()
    params = {'learning_rate': [0.01, 0.03, 0.1, 0.3],
              'n_estimators': [50, 125, 300],
              'subsample': [0.5, 1.0],
              'max_depth': [1, 3, 10]}
    mdl = xgb.XGBClassifier()
    gs = GridSearchCV(mdl, params, cv=5, n_jobs=-1)
    gs.fit(X, y)
    print 'Best params:', gs.best_params_
    print 'Best score:', gs.best_score_

    mdl = gs.best_estimator_
    with open('data/xgb_model.pkl', 'w') as f:
        pickle.dump(mdl, f)


def xgb_model_cv():
    X, y = load()

    print 'Running Model'
    mdl = xgb.XGBClassifier()
    scores = cross_val_score(mdl, X, y, cv=5, scoring='f1', n_jobs=-1)
    print 'f1 =', np.mean(scores)
    scores = cross_val_score(mdl, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print 'accuracy =', np.mean(scores)
    scores = cross_val_score(mdl, X, y, cv=5, scoring='recall', n_jobs=-1)
    print 'recall =', np.mean(scores)


def nlp_model_cv():
    X, y = load_desc()
    vec = TfidfVectorizer()
    mdl = xgb.XGBClassifier()

    acc_scores = []
    f1_scores = []
    kf = KFold(len(X), n_folds=5)
    print 'KFolding'
    for train_index, test_index in kf:
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        mdl.fit(X_train_vec, y_train)
        pred = mdl.predict(X_test_vec)
        acc_scores.append(accuracy_score(y_test, pred))
        f1_scores.append(f1_score(y_test, pred))

    print 'f1 =', np.mean(f1_scores)
    print 'accuracy =', np.mean(acc_scores)


if __name__ == '__main__':
    xgb_model()
    # xgb_model_cv()
    # nlp_model_cv()
