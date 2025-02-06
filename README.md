# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Your project description here.

## Files and data description
Overview of the files and data present in the root directory.   

    * data  
        * bank_data.csv
    * images  
        * eda  
            - churn_distribution.png  
            - customer_age_distribution.png  
            - heatmap.png  
            - marital_status_distribution.png  
            - Total_Transaction_distribution.png  
        * results  
            - feature_importances.png  
            - logistic_result.png  
            - rf_result.png  
            - rfc_roc_curve_result.png  
            - roc_curve_result.png  
            - shap_summary.png  
    * logs  
        * churn_library.log
    * models  
        * logistic_model.pkl
        * rfc_model.pkl
    * churn_library.py  
    * churn_notebook.ipynb  
    * churn_script_logging_and_tests.py  
    * constants.py  
    * requirements_py3.10.txt  
    * README.md  

## File Description
    * bank_data.csv  - file containing the customer data
    * churn_library.py  - Module consisting of methods to train and generate the model and artifacts
    * churn_notebook.ipynb  - Interactive python notebook to develop the model and experiment
    * churn_script_logging_and_tests.py  - pytest code to test the churn_library.py module
    * constants.py  - constants used in the module library
    * requirements_py3.10.txt  - file containting the required libraries for the project



## Running Files

1. install the required libraries  
    `python -m pip install -r requirements_py3.10.txt`
 
2. running the code  
    its build and tested with python 3.10  
    `python churn_library.py`

3. formatting the code  
    `autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py`  
    `autopep8 --in-place --aggressive --aggressive churn_library.py`  

4. code analysis for programming errors  
    `pylint churn_library.py`  
    `pylint churn_script_logging_and_tests.py`  
