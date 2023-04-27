import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def churn_pie(train):
    '''
    This function shows a pie chart of the percentage of customers that have churned within the training data
    '''
    y = train.churn.value_counts(normalize=True)

    mylabels = ["Did Not Churn", "Did Churn"]

    plt.pie(y, labels = mylabels, autopct='%1.1f%%')

    plt.show() 

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
    print(f" chi^2 = {chi2} p = {p}")

def relationship_churn(train, graph_title, feature, target):
    '''
    This function will take the train, graph_title, feature, and target,
    and it will display a bargraph based on the information provided for the churn dataset 

    '''
    fig, ax =plt.subplots()
    plt.title(graph_title)
    sns.barplot(x=feature, y=target, data=train)
    population_churn_rate = train.churn.mean()

    tick_label = ["No", "Yes"]
    ax.set_xticklabels(tick_label)
    # sns.distplot(train)

    plt.axhline(population_churn_rate, label="Population Churn Rate")
    plt.legend()
    plt.show()


def relationship_churn2(train, graph_title, feature, target):
    '''
    This function will take the train, graph_title, feature, and target,
    and it will display a bargraph based on the information provided for the churn dataset 

    '''
    fig, ax =plt.subplots()
    plt.title(graph_title)
    sns.barplot(x=feature, y=target, data=train)
    population_churn_rate = train.churn.mean()

    tick_label = ["Female", "Male"]
    ax.set_xticklabels(tick_label)
    # sns.distplot(train)

    plt.axhline(population_churn_rate, label="Population Churn Rate")
    plt.legend()
    plt.show()

def tenure_ttest(churn_sample, no_churn_sample, null_hypothesis, alternative_hypothesis):
    '''
    This function uses the indenpendent t-test to calculate the churn and nochurn sample for telco off of the feature tenure
    '''
    t, p = stats.ttest_ind(churn_sample, no_churn_sample, equal_var=False)

    alpha = 0.05
    if p < alpha:
        print(f"Reject the null hypothesis: {null_hypothesis}")
        print(f"Sufficient evidence to move forward understanding that, {alternative_hypothesis}")
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    print(f"p = {p}")