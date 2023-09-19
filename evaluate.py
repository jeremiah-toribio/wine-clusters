import wrangle as w
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Array and Dataframes
import numpy as np
import pandas as pd
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Evaluation: Statistical Analysis
from scipy import stats
# Metrics
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


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

def check_correlation(feature, compare, a=.05):
    '''
    - Shapiro's to check distribution
    - After distribution is determined, use pearsons or spearman r 
    '''
    α = a
    sr,sp = stats.shapiro(feature)

    if sp > α:
        print(f"Normal distribution, using pearson r")
        r, p = stats.pearsonr(compare, feature)
        if p < α:
            return print(f"""Reject the null hypothesis. There is a linear correlation.
        Pearson’s r: {r:2f}
        P-value: {p}""")
        else:
            return print(f"""We fail to reject the null hypothesis that there is a linear correlation.
            Spearman’s r: {r:2f}
            P-value: {p}""")
    else:
        print(f"NOT a normal distribution, spearman r")
        r, p = stats.spearmanr(compare, feature)
        if p < α:
            return print(f"""Reject the null hypothesis. There is a linear correlation.
        Spearman's r: {r:2f}
        P-value: {p}""")
        else:
            return print(f"""We fail to reject the null hypothesis that there is a linear correlation.
            Pearson's r: {r:2f}
            P-value: {p}""")

def t_test(feature, compare):
    '''
    '''
    α = .05
    lv, p = stats.levene(*samples, center='median', proportiontocut=0.05)

    if p > α:
        print(f'Equal Variance with a p-score of:{p}')
        t_stat, p = stats.ttest_ind(feature,compare,equal_var=True)
    else:
        print(f'INNEQUAL Variance with a p-score of:{p}')
        t_stat, p = stats.ttest_ind(feature,compare,equal_var=False)

    return check_p(p)

def plot_residuals(train, x_train, y_train, yhat, baseline):
    '''
    '''

    residuals = x_train - yhat
    residuals_baseline = x_train - baseline

    sns.scatter(data = train,x = residuals, y = y_train)
    #plt.plot(residuals, y_train)

    return

def regression_errors(x_train, y_train, yhat, baseline='mean'):
    '''
    '''
    if baseline == 'mean':
        baseline = y_train.mean()
    elif baseline == 'median':
        baseline = y_train.median()
    else:
        raise ValueError(' Give a proper aggregation for baseline.')
    
    residuals = x_train - yhat
    residuals_baseline = x_train - baseline

    
    SSE =  (residuals ** 2).sum()
    SSE_baseline = (residuals_baseline ** 2).sum()
    
    print('SSE =', "{:.1f}".format(SSE))
    print("SSE Baseline =", "{:.1f}".format(SSE_baseline))
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')

    MSE = SSE /len(x_train)
    MSE_baseline = SSE_baseline/len(x_train)

    print(f'MSE = {MSE:.1f}')
    print(f"MSE baseline = {MSE_baseline:.1f}")
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')

    ESS = (([yhat] - y_train.mean())**2).sum()
    ESS_baseline = (([baseline] - y_train.mean())**2).sum()
    print("ESS = ", "{:.1f}".format(ESS))
    print("ESS baseline = ", "{:.1f}".format(ESS_baseline))
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')

    RMSE = MSE ** .5
    RMSE_baseline = MSE_baseline ** .5
    print("RMSE = ", "{:.1f}".format(RMSE))
    print("RMSE baseline = ", "{:.1f}".format(RMSE_baseline))
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')
    return SSE, SSE_baseline, MSE, MSE_baseline, ESS, ESS_baseline, RMSE, RMSE_baseline

