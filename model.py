import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# tree classifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# rainforest classifier
from sklearn.ensemble import RandomForestClassifier

# linear regession classifier
from sklearn.linear_model import LogisticRegression

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier

def create_x_y(train, validate, test, target):
    """
    This function creates x and y variables for either a decision tree or a random forest, 
    by using the unsplit df, target variable columns name and column to drop, for multiple columns that need to be 
    dropped create a list of the columns0
    The arguments taken in are train, validate, test, target, drop_col=[])
    The function returns x_train, y_train, x_validate, y_validate, x_test, y_test
    """
    # separates train target variable
    x_train = train.drop(columns=[target])
    y_train = train[target]
    # validate 
    x_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # test
    x_test = test.drop(columns=[target])
    y_test = test[target]
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test

def clf_accuracy(clf, x_train, y_train, x_validate, y_validate):
    '''
    This function provides a quick print output of the train and validation scores based on your classifier, for easy viewing.
    The function takes the following arguments: object name (clf), x_train, y_train, x_validate, y_validate
    '''
    print(f'''
    Accuracy of {clf} on training set: {round(clf.score(x_train, y_train), 3)}
    Accuracy of {clf} on validation set: {round(clf.score(x_validate, y_validate), 3)}
    ''')

def clf_predict(clf, x_train, y_train): 
    '''
    This function takes in the following arguments: clf, x_train, y_train
    Then uses the arguments to make predictions on the train observation and clf
    creating a matrix and a df of the matrix 
    '''   
    # make prediction on train obeservations
    y_pred = clf.predict(x_train)

    # create confusion matrix
    conf = confusion_matrix(y_train, y_pred)

    # nice dataframe with conf
    labels = sorted(y_train.unique())
    df = pd.DataFrame(conf,
                index=[str(label) + '_actual'for label in labels],
                columns=[str(label) + '_predict'for label in labels])
    return df

def clf_test(clf, x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    ! WARNING!: Only use this for your final model 
    This function provides a quick print output of the baseling accuracy train, validation, test scores based on your classifier, for easy viewing.
    The function takes the following arguments: object name (clf), x_train, y_train, x_validate, y_validate, x_test, y_test
    '''
    print(f'''
    Baseline Accuracy: {round(baseline_accuracy, 3)}
    Accuracy of {clf} on training set: {round(clf.score(x_train, y_train), 3)}
    Accuracy of {clf} on validation set: {round(clf.score(x_validate, y_validate), 3)}
    Accuracy of {clf} on test set: {round(clf.score(x_test, y_test), 3)}
    ''')


####################################### Decision Tree model functions

def best_tree(x_train, y_train, x_validate, y_validate):
    tree = DecisionTreeClassifier(max_depth=5, random_state=3)

    # model.fit(x, y)
    tree = tree.fit(x_train, y_train)

    print(f'''
    Accuracy of Decision Tree classifier on training set: {round(tree.score(x_train, y_train),3)}
    Accuracy of Decision Tree classifier on validation set: {round(tree.score(x_validate, y_validate),3)}
    ''')

def dt_predict(tree, x_train, y_train): 
    '''
    This function takes in the following arguments: tree, x_train, y_train
    Then uses the arguments to make predictions on the train observation,
    creating a matrix and a df of the matrix 
    '''   
    # make prediction on train obeservations
    y_pred = tree.predict(x_train)

    # create confusion matrix
    conf = confusion_matrix(y_train, y_pred)

    # nice dataframe with conf
    labels = sorted(y_train.unique())
    df = pd.DataFrame(conf,
                index=[str(label) + '_actual'for label in labels],
                columns=[str(label) + '_predict'for label in labels])
    return df


def depth_check(x_train, y_train, x_validate, y_validate):
    scores_all = []
    for x in range(1,20):

        tree = DecisionTreeClassifier(max_depth=x, random_state=3)
        tree.fit(x_train, y_train)
        train_acc = tree.score(x_train, y_train)

        #evaluate on validate
        val_acc = tree.score(x_validate, y_validate)

        scores_all.append([x, round(train_acc, 6), round(val_acc, 6)])
        
    scores = pd.DataFrame(scores_all, columns=['max_depth','train_acc','val_acc'])
    
    scores['diff'] = round(scores.train_acc - scores.val_acc, 6)
    return scores


    

def dt_depth_graph(x_train, y_train, x_validate, y_validate):
    '''
    This function takes in: x_train, y_train, x_validate, y_validate
    Which then runs through a range of (1,20) to help determine the best parameter for max_depth.
    It outputs a visual graph and a table with the accuracy results and difference in score for better viewing. 
    '''

    # create scoring table
    scores = depth_check(x_train, y_train, x_validate, y_validate)


    # creating graph
    fig = plt.figure(figsize=(15,8))
    fig.suptitle('how does the accuracy change with max depth on train and validate?')
    
    ax1 = fig.add_subplot(121)
    

    ax1.plot(scores.max_depth, scores.train_acc, label='train', marker='o')
    ax1.plot(scores.max_depth, scores.val_acc, label='validation', marker='o')

    ax1.legend()

#     Create visual table
    ax2 = fig.add_subplot(122)
    font_size=9
    bbox=[0, 0, 1, 1]
    ax2.axis('off')
    mpl_table = ax2.table(cellText = scores.values, rowLabels = scores.index, bbox=bbox, colLabels=scores.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    



######################################## Random Forest model functions 

def best_forest(x_train, y_train, x_validate, y_validate):
    rf = RandomForestClassifier(random_state=3, min_samples_leaf=8, max_depth=7)

    # model.fit(x, y)
    rf = rf.fit(x_train, y_train)

    print(f'''
    Accuracy of Decision Tree classifier on training set: {round(rf.score(x_train, y_train),3)}
    Accuracy of Decision Tree classifier on validation set: {round(rf.score(x_validate, y_validate),3)}
    ''')

def rf_predict(rf, x_train, y_train): 
    '''
    This function takes in the following arguments: tree, x_train, y_train
    Then uses the arguments to make predictions on the train observation,
    creating a matrix and a df of the matrix 
    '''   
    # make prediction on train obeservations
    y_pred = rf.predict(x_train)

    # create confusion matrix
    conf = confusion_matrix(y_train, y_pred)

    # nice dataframe with conf
    labels = sorted(y_train.unique())
    df = pd.DataFrame(conf,
                index=[str(label) + '_actual'for label in labels],
                columns=[str(label) + '_predict'for label in labels])
    return df



def leaf_check(x_train, y_train, x_validate, y_validate):
    '''
    This function takes in: x_train, y_train, x_validate, y_validate
    Which then runs through a range of (1,11)-min_samples_leaf and descending (1,11) for max_depth to help determine the best parameters
    '''
    scores_all = []

    for x in range(1,11):

        #make it
        rf = RandomForestClassifier(random_state=3, min_samples_leaf=x, max_depth=11-x)
        #fit it
        rf.fit(x_train, y_train)
        #transform it
        train_acc = rf.score(x_train, y_train)

        #evaluate on my validate data
        val_acc = rf.score(x_validate, y_validate)

        scores_all.append([x, 11-x, round(train_acc, 4), round(val_acc, 4)])

    scores_df = pd.DataFrame(scores_all, columns=['min_samples_leaf','max_depth','train_acc','val_acc'])
    scores_df['difference'] = round(scores_df.train_acc - scores_df.val_acc, 3)
    return scores_df


def rf_leaf_graph(x_train, y_train, x_validate, y_validate):
    '''
    This function takes in: x_train, y_train, x_validate, y_validate
    Which then runs through a range of (1,11)-min_samples_leaf and descending (1,11) for max_depth to help determine the best parameters
    It outputs a visual graph and a table with the accuracy results and difference in score for better viewing. 
    '''

    # create scoring table
    scores = leaf_check(x_train, y_train, x_validate, y_validate)


    # creating graph
    fig = plt.figure(figsize=(15,8))
    fig.suptitle('how does the accuracy change with (min_samples_leaf asc. 1,11) and (max depth desc. 1,11) on train and validate?')
    
    ax1 = fig.add_subplot(121)
    

    ax1.plot(scores.max_depth, scores.train_acc, label='train', marker='o')
    ax1.plot(scores.max_depth, scores.val_acc, label='validation', marker='o')
    ax1.set_xlabel('max depth and min leaf sample')
    ax1.set_ylabel('accuracy')
    ax1.legend()

#     Create visual table
    ax2 = fig.add_subplot(122)
    font_size=9
    bbox=[0, 0, 1, 1]
    ax2.axis('off')
    mpl_table = ax2.table(cellText = scores.values, rowLabels = scores.index, bbox=bbox, colLabels=scores.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

#################################### Logistic Regression model functions 
def logit_accuracy(x_train, y_train, x_validate, y_validate):
    '''
    This function provides a quick print output of the train and validation scores based on the logisitics regression, for easy viewing.
    The function takes the following arguments: logit, x_train, y_train, x_validate, y_validate
    '''
    logit = LogisticRegression()

    # model fit 
    logit.fit(x_train, y_train)
    print(f'''

    Accuracy of Logistic Regression on training set: {round(logit.score(x_train, y_train),3)}
    Accuracy of Logistic Regression on validation set: {round(logit.score(x_validate, y_validate),3)}
    ''')


####################### KNN model functions
def best_knn(x_train, y_train, x_validate, y_validate):
    '''
    This function provides a quick print output of the train and validation scores based on the KNN model, for easy viewing.
    The function takes the following arguments: logit, x_train, y_train, x_validate, y_validate
    '''
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(x_train, y_train)
    print(f'''
    Accuracy of KNN on training set: {round(knn.score(x_train, y_train),3)}
    Accuracy of KNN on validation set: {round(knn.score(x_validate, y_validate),3)}
    ''')
