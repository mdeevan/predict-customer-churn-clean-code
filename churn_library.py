'''
identify credit card customers who are most likely to churn.
The library is created after experimenting in the the python notbook.

Author: Muhammad Naveed
Created On : Feb 5th, 2025
'''

# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.metrics import RocCurveDisplay, classification_report

import constants

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class SplitData():
    '''
    class to hold the splitted data, to help reduce the
    number of parameters when passing around
    '''

    def __init__(self, X_train, X_test, y_train, y_test):
        '''
        class to hold the splitted dataset for training and test
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def import_data(filename="bank_data.csv"):
    '''
    returns dataframe for the csv found at path

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    data = pd.read_csv(os.path.join(constants.DATA_FOLDER, filename))

    return data


def get_path(folder, filename):
    '''
    form a path by concatenating folder and filename
    INPUT:
        folder: str : folder name
        filename: str : filename
    OUTPUT:
        concatenated path and filename
    '''
    return os.path.join(folder, filename)


def perform_eda(df, path=constants.EDA_IMAGES_FOLDER):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            path: path to store the eda output

    output:
            None
    '''

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(get_path(path, "churn_distribution.png"))
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(get_path(path, "customer_age_distribution.png"))
    plt.close()

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(get_path(path, "marital_status_distribution.png"))
    plt.close()

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct'])
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(get_path(path, "Total_Transaction_distribution.png"))
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(get_path(path, "heatmap.png"))
    plt.close()

    return df


def encoder_helper(df, category_lst=None, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    if category_lst is None:
        category_lst = constants.CAT_COLUMNS

    for col in category_lst:
        lst = []
        grp = df.groupby(col).mean(numeric_only=True)['Churn']

        for val in df[col]:
            lst.append(grp.loc[val])

        df[f"{col}_{response}"] = lst

    return df


def perform_feature_engineering(df, keep_cols=None):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument
        that could be used for naming variables or index y column]

    output:
        X : the original data set
        y: the original target data set
        split_data comprising of following
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''

    if keep_cols is None:
        keep_cols = constants.KEEP_COLS

    X = df[keep_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)

    split_data = SplitData(X_train, X_test, y_train, y_test)
    return X, y, split_data


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                path=constants.RESULT_FOLDER):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(get_path(path, "rf_result.png"))

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(get_path(path, "logistic_result.png"))


def feature_importance_plot(model, X_data, path=constants.RESULT_FOLDER):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(get_path(path, "feature_importances.png"))


def train_models(split_data,
                 model_folder=constants.MODEL_FOLDER,
                 result_folder=constants.RESULT_FOLDER):
    '''
    train, store model results: images + scores, and store models
    input:
        split_data, which has the following attributes
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(split_data.X_train, split_data.y_train)

    lrc.fit(split_data.X_train, split_data.y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(split_data.X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(split_data.X_test)

    y_train_preds_lr = lrc.predict(split_data.X_train)
    y_test_preds_lr = lrc.predict(split_data.X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(split_data.y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(split_data.y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(split_data.y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(split_data.y_train, y_train_preds_lr))

    # save best model
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            model_folder,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(model_folder, 'logistic_model.pkl'))

    # lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    lrc_plot = RocCurveDisplay.from_estimator(lrc,
                                              split_data.X_test,
                                              split_data.y_test)

    plt.figure(figsize=(15, 8))

    ax = plt.gca()
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(get_path(result_folder, "roc_curve_result.png"))

    # rfc_disp = plot_roc_curve(cv_rfc, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_,
                                              split_data.X_test,
                                              split_data.y_test,
                                              ax=ax,
                                              alpha=0.8)

    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.savefig(get_path(result_folder, "rfc_roc_curve_result.png"))

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(split_data.X_test)
    shap.summary_plot(shap_values, split_data.X_test, plot_type="bar")
    plt.savefig(get_path(result_folder, "shap_summary.png"))

    return (y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
            cv_rfc)


if __name__ == "__main__":

    def execute_module():
        '''
        to create a local scope, to avoid creating a global variable
        which potentially conflicts with the local variables
        within other functions and methods

        '''
        df = import_data("bank_data.csv")

        df = perform_eda(df, constants.EDA_IMAGES_FOLDER)
        df = encoder_helper(df, constants.CAT_COLUMNS, 'Churn')

        (X_data, y_data, split_data) = \
            perform_feature_engineering(df, constants.KEEP_COLS)

        y_train_preds_lr, y_train_preds_rf, \
            y_test_preds_lr, y_test_preds_rf, \
            model = train_models(split_data)

        classification_report_image(split_data.y_train,
                                    split_data.y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf)

        feature_importance_plot(model, X_data, constants.RESULT_FOLDER)

    execute_module()
