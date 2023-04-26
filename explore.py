import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def cross_function(train, target_variable, feature_variable, null_hypothesis, alternative_hypothesis, alpha=0.05):
    '''
    This function will take the train, target_variable, feature_variable, null_hypothesis, alternative_hypothesis, alpha=0.05
    and print the results and the p-value
    '''
    observed = pd.crosstab(train[target_variable], train[feature_variable])

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    if p < alpha:
        print(f"Reject the null hypothesis: {null_hypothesis}")
        print(f"Sufficient evidence to move forward understanding that, {alternative_hypothesis}")
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    return p

def relationship_churn(train, graph_title, feature, target):
    '''
    This function will take the train, graph_title, feature, and target,
    and it will display a bargraph based on the information provided for the churn dataset 

    '''
    fig, ax =plt.subplots()
    plt.title(graph_title)
    sns.barplot(x=feature, y=target, data=train)
    population_churn_rate = train.churn.mean()

    tick_label = ["Did Not Churn", "Did Churn"]
    ax.set_xticklabels(tick_label)
    # sns.distplot(train)

    plt.axhline(population_churn_rate, label="Population Churn Rate")
    plt.legend()
    plt.show()