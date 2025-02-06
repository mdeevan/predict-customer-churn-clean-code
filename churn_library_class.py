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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import RocCurveDisplay, classification_report

import constants

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class CustomerChurn():
    def __init__(self,  filename="bank_data.csv", nrows=None):
        self.filename = filename
        self.df = pd.DataFrame()
        self.nrows = nrows
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_data = None
        self.y_data = None
        self.y_train_preds_rf = None
        self.y_test_preds_rf = None
        self.y_train_preds_lr = None
        self.y_test_preds_lr = None
        self.model = None

    def __get_path(self, folder, filename):
        '''
        form a path by concatenating folder and filename
        INPUT:
            folder: str : folder name
            filename: str : filename
        OUTPUT:
            concatenated path and filename
        '''
        return os.path.join(folder, filename)

    def _import_data(self, filename="bank_data.csv"):
        '''
        returns dataframe for the csv found at path

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''

        self.df = pd.read_csv(os.path.join(constants.DATA_FOLDER, self.filename), 
                              nrows=self.nrows)

    def _perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        Class variables as input
                df: pandas dataframe
                path: path to store the eda output

        output:
                None
        '''

        path = constants.EDA_IMAGES_FOLDER

        self.df['Churn'] = self.df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    
        plt.figure(figsize=(20, 10))
        self.df['Churn'].hist()
        plt.savefig(self.__get_path(path, "churn_distribution.png"))
        plt.close()
    
        plt.figure(figsize=(20, 10))
        self.df['Customer_Age'].hist()
        plt.savefig(self.__get_path(path, "customer_age_distribution.png"))
        plt.close()
    
        plt.figure(figsize=(20, 10))
        self.df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(self.__get_path(path, "marital_status_distribution.png"))
        plt.close()
    
        plt.figure(figsize=(20, 10))
        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct'])
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
        # using a kernel density estimate
        sns.histplot(self.df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(self.__get_path(path, "Total_Transaction_distribution.png"))
        plt.close()
    
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self.df.corr(numeric_only=True),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig(self.__get_path(path, "heatmap.png"))
        plt.close()
    
    def _encoder_helper(self, category_lst=None, response='Churn'):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook
    
        input:
                df: pandas dataframe (class variable, to be used)
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
            grp = self.df.groupby(col).mean(numeric_only=True)['Churn']
    
            for val in self.df[col]:
                lst.append(grp.loc[val])
    
            self.df[f"{col}_{response}"] = lst
    

    def _perform_feature_engineering(self, keep_cols=None):
        '''
        input:
            df: pandas dataframe (instance variable)
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
    
        self.X_data = self.df[keep_cols]
        self.y_data = self. df['Churn']
    
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X_data, self.y_data, test_size=0.3, random_state=42)

    def _train_models(self,
                     model_folder=constants.MODEL_FOLDER,
                     result_folder=constants.RESULT_FOLDER):
        '''
        train, store model results: images + scores, and store models
        input:
            instances variables
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
        cv_rfc.fit(self.X_train, self.y_train)
        self.model = cv_rfc
    
        lrc.fit(self.X_train, self.y_train)
    
        self.y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        self.y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)
    
        self.y_train_preds_lr = lrc.predict(self.X_train)
        self.y_test_preds_lr = lrc.predict(self.X_test)
    
        # scores
        print('random forest results')
        print('test results')
        print(classification_report(self.y_test, self.y_test_preds_rf))
        print('train results')
        print(classification_report(self.y_train, self.y_train_preds_rf))
    
        print('logistic regression results')
        print('test results')
        print(classification_report(self.y_test, self.y_test_preds_lr))
        print('train results')
        print(classification_report(self.y_train, self.y_train_preds_lr))
    
        # save best model
        joblib.dump(
            cv_rfc.best_estimator_,
            self.__get_path(model_folder, 'rfc_model.pkl'))
        joblib.dump(lrc, self.__get_path(model_folder, 'logistic_model.pkl'))
    
        # lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        lrc_plot = RocCurveDisplay.from_estimator(lrc,
                                                  self.X_test,
                                                  self.y_test)
    
        plt.figure(figsize=(15, 8))
    
        ax = plt.gca()
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(self.__get_path(result_folder, "roc_curve_result.png"))
    
        # rfc_disp = plot_roc_curve(cv_rfc, X_test, y_test, ax=ax, alpha=0.8)
        rfc_disp = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_,
                                                  self.X_test,
                                                  self.y_test,
                                                  ax=ax,
                                                  alpha=0.8)
    
        rfc_disp.plot(ax=ax, alpha=0.8)
        plt.savefig(self.__get_path(result_folder, "rfc_roc_curve_result.png"))
    
        explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")
        plt.savefig(self.__get_path(result_folder, "shap_summary.png"))
    
    def _classification_report_image(self,
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

        plt.text(0.01, 1.25, str('Random Forest Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_test, 
                                                       self.y_test_preds_rf)),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Random Forest Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_train,
                                                      self.y_train_preds_rf)),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(self.__get_path(path, "rf_result.png"))
    
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_train, 
                                                       self.y_train_preds_lr)), 
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Logistic Regression Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_test, 
                                                      self.y_test_preds_lr)),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(self.__get_path(path, "logistic_result.png"))

    def _feature_importance_plot(self, path=constants.RESULT_FOLDER):
        '''
        creates and stores the feature importances in pth
        input:
                model: instance variable
                X_data: instance variable
                output_pth: path to store the figure
    
        output:
                 None
        '''
        # Calculate feature importances
        importances = self.model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
    
        # Rearrange feature names so they match the sorted feature importances
        names = [self.X_data.columns[i] for i in indices]
    
        # Create plot
        plt.figure(figsize=(20, 5))
    
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
    
        # Add bars
        plt.bar(range(self.X_data.shape[1]), importances[indices])
    
        # Add feature names as x-axis labels
        plt.xticks(range(self.X_data.shape[1]), names, rotation=90)
        plt.savefig(self.__get_path(path, "feature_importances.png"))

    def execute(self):
        self._import_data()
        print("data imported")
        self._perform_eda()
        print('eda performed')
        self._encoder_helper()
        print('encoder helper done')
        self._perform_feature_engineering()
        print('featured engineered')
        self._train_models()
        print('model trained')
        self._classification_report_image()
        print('classification report done')
        self._feature_importance_plot()
        print('feature importance done')


if __name__ == "__main__":

    customer_churn = CustomerChurn() 
    print("object created, lets' execute")
    customer_churn.execute()
    