# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# score(X, y, sample_weight=None)[source]
# Return the coefficient of determination R^2 of the prediction.

# The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.


def lasso_cv(X,y,k):
    #####
    ## X: CP data [perts/samples, features]
    ## y: lm gene expression value [perts/samples, feature value]
    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn import metrics

    # build sklearn model
    clf = linear_model.Lasso(alpha=0.1,max_iter=10000)

    # Perform 6-fold cross validation
    scores = cross_val_score(clf, X, y, cv=k)
    return scores