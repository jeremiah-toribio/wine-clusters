# Array and Dataframes
import numpy as np
import pandas as pd
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# Decision Tree
from sklearn.tree import DecisionTreeClassifier as dt, plot_tree, export_text
# Logistic Regression
from sklearn.linear_model import LogisticRegression as log
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
    # Linear: Polynomial
#from sklearn.preprocessing import PolynomialFeatures
    # Feature Selection
from sklearn.feature_selection import SelectKBest, RFE, f_regression


def check_p(p):
    '''
    checks p value to see association to a, depending on outcome will print
    relative statement
    '''
    α = .05
    if p < α:
        return print(f'We can reject the null hypothesis with a p-score of:',{p})
    else:
        return print(f'We fail to reject the null hypothesis with a p-score of:',{p})


def get_classification_report(x_test, y_pred):
    '''
    Returns classification report as a dataframe.
    '''
    report = classification_report(x_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report


def log_regression_compiled(x_train, y_train, x_validate, y_validate):
    '''
    Generates the logistic regression sklearn model.
    Finds the best fit C parameter using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    '''
    # Parameters defined for GridSearch, train model
    param_grid_L1 = {'penalty': ['l1', 'l2'], 'C': np.arange(.1,4,.1)}
    logreg_tuned = log(solver='saga', max_iter=150) # solver='saga', max_iter=150
    logreg_tuned_gs = GridSearchCV(logreg_tuned, param_grid_L1, cv=5)
    logreg_tuned_gs.fit(x_train,y_train)

    # Predictions based on trained model
    y_predictions_log_tuned = logreg_tuned_gs.predict(x_validate)
    y_predictions_log_prob_tuned = logreg_tuned_gs.predict_proba(x_validate)

    # Output best C parameter
    print(f'Best fit "C" parameter (Determined by GridSearchCV): {logreg_tuned_gs.best_params_["C"]}')

    # tree object
    logit = log(C=logreg_tuned_gs.best_params_["C"], random_state=4343)
    # fit
    logit.fit(x_train,y_train)
    # predict
    model_prediction = logit.predict(x_train)
    model_prediction_test = logit.predict(x_validate)

    print(accuracy_score(y_validate,model_prediction_test))
    # # generate metrics
    # TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    # get_classification_report(y_train,model_prediction)
    # metrics(TN, FP, FN, TP)
    # # test metrics
    # TN, FP, FN, TP = confusion_matrix(y_validate, model_prediction_test).ravel()
    # get_classification_report(y_validate,model_prediction_test)

    return


def knn_compiled(x_train, y_train, x_validate, y_validate, weights='uniform'):
    '''
    Generates the K Nearest Neighbors SKLearn model.
    Finds the best fit C parameter using GridSearchCV by SKLearn
    Scans from n_neighbors = 1-30
    x_train = features
    y_train = target
    weights = 'decide based on distribution'
    '''
 # Parameters defined for GridSearch, train model
    param_grid = {'n_neighbors': np.arange(2,20)}
    knn = KNeighborsClassifier(n_neighbors=28,weights=weights)
    knn_cv = GridSearchCV(knn,param_grid,cv=5, scoring ='accuracy',return_train_score=True)
    knn_cv.fit(x_train,y_train)
    
    # Predictions based on trained model
    grid_pred = knn_cv.predict(x_validate)
    print(f'Best fit "n_neighbors" parameter (Determined by GridSearchCV): {knn_cv.best_params_["n_neighbors"]}',\
      '\n--------------------------------------')
    
    print(f' {knn_cv.best_score_}')
    print(accuracy_score(y_validate,grid_pred))
    # knn object
    # knn = KNeighborsClassifier(n_neighbors=knn_cv, weights=weights)

    # # fit
    # knn.fit(x_train,y_train)
    
    # # predict
    # model_prediction = knn.predict(x_train)
    # model_prediction_test = knn.predict(x_validate)
    # # generate metrics
    # TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    # print(f'Train Class Report & Metrics:\
    #   \n---------------------------------------')
    # print(f'{classification_report(y_train,model_prediction)}')
    # print(f'{metrics(TN, FP, FN, TP)}','\n')
    # # test metrics
    # TN, FP, FN, TP = confusion_matrix(y_validate, model_prediction_test).ravel()
    # print(f'Test Classificiation Report & Metrics:\
    #   \n--------------------------------------')
    # print(f'{classification_report(y_validate,model_prediction_test)}')
    # print(f'{metrics(TN, FP, FN, TP)}')
    return

def rfe(x_train_scaled, x_train, y_train, model, k=2):
    '''
    Input type of model to optimize for

    model = 'ols' (ordinary least squares); 
    '''
    # create the rfe object, indicating the ML object (lr) and the number of features I want to end up with. 
    if model == 'ols':
        model = lr()
    elif model == 'lassolars':
        model = LassoLars()
    elif model.__contains__('glm') == True:
        if model == ('glm0'):
            model = TweedieRegressor(power=0)
        elif model == ('glm1'):
            model = TweedieRegressor(power=1)
        elif model == ('glm2'):
            model = TweedieRegressor(power=2)
        elif model == ('glm3'):
            model = TweedieRegressor(power=3)
    else:
        raise ValueError('Select a valid model.')
    
    rfe = RFE(estimator=model, n_features_to_select=k)

    # fit the data using RFE
    rfe.fit(x_train_scaled,y_train) 

    # get list of the column names. 
    rfe_feature = x_train.columns[rfe.support_].tolist()

    return rfe_feature

def select_kbest(x_train_scaled,x_train, y_train, k=2):

    # parameters: f_regression stats test, give me all features - normally in
    f_selector = SelectKBest(f_regression, k=k)#k='all')
    # find the all X's correlated with y
    f_selector.fit(x_train_scaled, y_train)

    # boolean mask of whether the column was selected or not
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = x_train.iloc[:,feature_mask].columns.tolist()

    return f_feature

