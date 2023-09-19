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
    For independent feature testing
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

def chi2_test(col1, col2, a=.05):
    '''
    NOTE: Requires stats from scipy in order to function
    A faster way to test two columns desired for cat vs. cat statistical analysis.

    Default α is set to .05.

    Outputs crosstab and respective chi2 relative metrics.
    '''
    observed = pd.crosstab(col1, col2)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    if p < a:
        print(f'We can reject the null hypothesis with a p-score of:',{p})
    else:
        print(f'We fail to reject the null hypothesis with a p-score of:',{p})
    
    return observed
