import cPickle as pickle 
from clean_data import clean, clean_desc



def predict(test, xgb_model):

    # with open('vec_model.pkl') as f:
    #     vec_model = pickle.load(f)

    X_xgb = clean(test, isTrain=False, isjson=True)
    # X_vec = clean_desc(test_file, isTrain=False)

    xgb_preds = xgb_model.predict_proba(X_xgb)[:,1]
    # vec_preds = vec_model.predict_proba(X_vec)[:,1]

    # preds = (xgb_preds + vec_preds) / 2

    # return X_xgb, X_vec, preds

    return X_xgb, xgb_preds
