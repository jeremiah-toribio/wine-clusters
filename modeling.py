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
from sklearn.linear_model import LogisticRegression as lr
# KNN
from sklearn.neighbors import KNeighborsClassifier


def eval_metrics(tp,tn,fp,fn):
        '''Input:
        True positive(tp),
        True negative (tn),
        False positive (fp),
        False negative (fn)

        Reminder:
        false pos true neg
        false neg true pos
        '''
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        return print(f'''Accuracy is: {accuracy}\nPrecision is: {precision}\nRecall is: {recall}''')





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


def metrics(TN,FP,FN,TP):
    '''
    True positive(TP),
        True negative (TN),
        False positive (FP),
        False negative (FN)

        Reminder:
        false pos true neg
        false neg true pos
    '''
    combined = (TP + TN + FP + FN)

    accuracy = (TP + TN) / combined

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)


    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN

    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")


def decision_tree_compiled(x_train, y_train, x_validate, y_validate, df, plot=True):
    '''
    x_train = features
    y_train = target
    df is used to generate the values in churn in this case.
    Optional tree visualization. Default True.
    '''
    # tree object
    clf = dt(max_depth=3,random_state=4343)
    # fit
    clf.fit(x_train, y_train)
    # predict
    model_prediction = clf.predict(x_train)

    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    get_classification_report(y_train,model_prediction)
    metrics(TN, FP, FN, TP)

    # plot Tree
    if plot == True:
        labels = list(df['churn'].astype(str))
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=500)
        tree = plot_tree(clf, feature_names=x_train.columns.to_list(), class_names=labels,filled=True)
        plt.show()
    else:
        None
    return


def log_regression_compiled(x_train, y_train, x_validate, y_validate):
    '''
    Generates the logistic regression sklearn model.
    Finds the best fit C parameter using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    '''
    # Parameters defined for GridSearch, train model
    param_grid_L1 = {'penalty': ['l1', 'l2'], 'C': np.arange(.1,5,.1)}
    logreg_tuned = lr(solver='saga', max_iter=500)
    logreg_tuned_gs = GridSearchCV(logreg_tuned, param_grid_L1, cv=5)
    logreg_tuned_gs.fit(x_train,y_train)

    # Predictions based on trained model
    y_predictions_lr_tuned = logreg_tuned_gs.predict(x_validate)
    y_predictions_lr_prob_tuned = logreg_tuned_gs.predict_proba(x_validate)

    # Output best C parameter
    print(f'Best fit "C" parameter (Determined by GridSearchCV): {logreg_tuned_gs.best_params_["C"]}')

    # tree object
    logit = lr(C=logreg_tuned_gs.best_params_["C"], random_state=4343)
    # fit
    logit.fit(x_train,y_train)
    # predict
    model_prediction = logit.predict(x_train)
    model_prediction_test = logit.predict(x_validate)

    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    get_classification_report(y_train,model_prediction)
    metrics(TN, FP, FN, TP)
    # test metrics
    TN, FP, FN, TP = confusion_matrix(y_validate, model_prediction_test).ravel()
    get_classification_report(y_validate,model_prediction_test)

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
    param_grid = {'n_neighbors': np.arange(1,30)}
    knn = KNeighborsClassifier(n_neighbors=28,weights='uniform')
    knn_cv = GridSearchCV(knn,param_grid,cv=5)
    knn_cv.fit(x_train,y_train)
    
    # Predictions based on trained model
    y_pred_knn_tuned = knn_cv.predict(x_validate)
    print(f'Best fit "n_neighbors" parameter (Determined by GridSearchCV): {knn_cv.best_params_["n_neighbors"]}',\
      '\n--------------------------------------')

    # knn object
    knn = KNeighborsClassifier(n_neighbors=knn_cv, weights=weights)

    # fit
    knn.fit(x_train,y_train)
    
    # predict
    model_prediction = knn.predict(x_train)
    model_prediction_test = knn.predict(x_validate)
    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    print(f'Train Class Report & Metrics:\
      \n---------------------------------------')
    print(f'{classification_report(y_train,model_prediction)}')
    print(f'{metrics(TN, FP, FN, TP)}','\n')
    # test metrics
    TN, FP, FN, TP = confusion_matrix(y_validate, model_prediction_test).ravel()
    print(f'Test Classificiation Report & Metrics:\
      \n--------------------------------------')
    print(f'{classification_report(y_validate,model_prediction_test)}')
    print(f'{metrics(TN, FP, FN, TP)}')
    return
