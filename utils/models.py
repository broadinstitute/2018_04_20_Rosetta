

def lasso_cv(X,y,k):
    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn import metrics

    # build sklearn model
    clf = linear_model.Lasso(alpha=0.1)

    # Perform 6-fold cross validation
    scores = cross_val_score(model, df, y, cv=k)
    return scores