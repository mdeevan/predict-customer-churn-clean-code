'''
identify credit card customers who are most likely to churn.
the testing script is to test the functionalities defined in churn_library.py

Author: Muhammad Naveed
Created On : Feb 5th, 2025
'''

import os
import logging
import pytest

import constants
import churn_library as cls


# https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    force=True,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


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


@pytest.fixture
def df():
    '''
    import the data and and return only the first 100 rows,
    this helps speed up the testing

    OUTPUT:
        df:  dataframe for use in rest of the testing
    '''

    data = cls.import_data("bank_data.csv")
    print(type(data))
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return data[:50]


def test_import_data():
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    INPUT:
        none
    OUTPUT:
        none
    '''

    try:
        data = cls.import_data("bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't \
            appear to have rows and columns")
        raise err


@pytest.mark.use_fixtures(df_in=[df])
def test_perform_eda(df_in, tmp_path):
    '''
    test perform eda function
    INPUT:
        df: Fixture:  retrieve the dataframe from the Fixture defined earlier
        tmp_path: Fixture: retrieve the temporary path for storing the files
        created during the test execution
    OUTPUT:
        None
    '''
    try:
        cls.perform_eda(df_in, tmp_path)
        logging.info('perform_eda: SUCCESS')
    except AssertionError as err:
        logging.error("perofrm_eda: FAILURE")
        raise err

# Define the names of the image files created by the procedure and
# validate the existence of each of these in a loop
    files = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "Total_Transaction_distribution.png",
        "heatmap.png"]

    for filename in files:
        try:
            assert os.path.exists(os.path.join(tmp_path, filename))
            logging.info("file path %s exists: SUCCESS", filename)
        except AssertionError as err:
            logging.error("file path % exists: FAILURE", filename)
            raise err


# @pytest.mark.parametrize("filename", ["churn_distribution.png", \
# "customer_age_distribution.png", "marital_status_distribution.png", \
# "Total_Transaction_distribution.png", "heatmap.png" ])

# def test_checkFile(tmp_path, filename):
#     try:
#         assert os.path.exists(os.path.join(tmp_path, filename)) #== True
#         logging.error("file path exists: SUCCESS")
#     except AssertionError as err:
#         logging.error("file path exists: FAILURE")
#         raise err

@pytest.mark.use_fixtures(df_in=[df])
def test_encoder_helper(df_in):
    '''
    test encoder helper
    INPUT:
        df: Fixture: to Retrieve the data to use for the test
    OUTPUT:
        data: dataframe : Returns the dataframe modified via
        encoder-helper procedure
    '''

    try:
        data = cls.encoder_helper(
            df_in, constants.CAT_COLUMNS, constants.SUFFIX)
        logging.info("encoder helper : SUCCESS")
    except AssertionError as err:
        logging.error("encoder helper : FAILURE")
        raise err

#   Loop through all the category columns defined in the constants file
#   and check if these are created
#   column existence is checked via testing the shape
    for col in constants.CAT_COLUMNS:
        try:
            col_name = f"{col}_{constants.SUFFIX}"
            if data[col_name].shape[0] > 0:
                logging.info("%s successfully created : SUCCESS", col_name)

        except KeyError as err:
            logging.error("%s not created : FAILURE", col_name)
            raise err

    # return data


@pytest.mark.use_fixtures(df_in=[df])
def test_perform_feature_engineering(df_in):
    '''
    test perform_feature_engineering
    '''
    data = cls.encoder_helper(df_in, constants.CAT_COLUMNS, constants.SUFFIX)

    try:
        # X_data, y_data, X_train, X_test, y_train, y_test = cls.perform_feature_engineering(data)

        _, _, split_data = cls.perform_feature_engineering(data)
        logging.info("perform feature engineering: SUCCESS")

    except (AssertionError, KeyError, ValueError) as err:
        logging.error("perform feature engineering: FAILURE")
        raise err

    lst_obj = [split_data.X_train, split_data.X_test,
               split_data.y_train, split_data.y_test]
    lst_str = ["X_train", "X_test", "y_train", "y_test"]

    for i, obj in enumerate(lst_obj):
        try:
            if obj.shape[0] > 0:
                logging.info("%s has a shape of {obj.shape}", lst_str[i])

        except (AssertionError, KeyError) as err:
            logging.error("%s failed to be created", lst_str[i])
            raise err


@pytest.mark.use_fixtures(df_in=[df])
def test_train_models(df_in, tmp_path):
    '''
    test train_models
    INPUT:
        df: Fixture:  retrieve the dataframe from the Fixture defined earlier
        tmp_path: Fixture: retrieve the temporary path for storing the files
        created during the test execution
    OUTPUT:
        None

    '''
#   call the encode to update the dataframe, following by feature engineering
#   these two methods readies the dataframe for training

    data = cls.encoder_helper(df_in, constants.CAT_COLUMNS, constants.SUFFIX)
    # X_data, y_data, X_train, X_test, y_train, y_test = cls.perform_feature_engineering(data)
    _, _, split_data = cls.perform_feature_engineering(data)

    try:
        # y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf,
        # model = cls.train_models(X_train, X_test, y_train, y_test,
        # tmp_path, tmp_path)

        (_, _, _, _, model) = cls.train_models(split_data, tmp_path, tmp_path)
        if model is not None:
            logging.info("Training model : SUCCESS")
        else:
            logging.info("Training model : FAILURE")
    except AssertionError as err:
        logging.error("Training model: FAILURE")
        raise err


@pytest.mark.use_fixtures(df_in=[df])
def test_feature_importance_plot(df_in, tmp_path):
    '''
    test feature importance plot
    INPUT:
        df: Fixture:  retrieve the dataframe from the Fixture defined earlier
        tmp_path: Fixture: retrieve the temporary path for storing the files
        created during the test execution
    OUTPUT:
        None

    '''
    data = cls.encoder_helper(df_in, constants.CAT_COLUMNS, constants.SUFFIX)
    # X_data, y_data, \
    #     X_train, X_test, \
    #     y_train, y_test = cls.perform_feature_engineering(data)
    X_data, _, split_data = cls.perform_feature_engineering(data)

    # (y_train_preds_lr,
    #  y_train_preds_rf,
    #  y_test_preds_lr,
    #  y_test_preds_rf,
    (_, _, _, _,
     model) = cls.train_models(split_data,
                               tmp_path,
                               tmp_path)

    try:
        cls.feature_importance_plot(model, X_data, tmp_path)
        logging.info("Feature Importance plot : SUCCESS")
    except AssertionError as err:
        logging.error("Feature Important plot: FAILURE")
        raise err

    filename = "feature_importances.png"
    try:
        assert os.path.exists(
            os.path.join(tmp_path, filename))
        logging.info("file path %s exists: SUCCESS", filename)
    except (AssertionError, FileNotFoundError) as err:
        logging.error("file path %s exists: FAILURE", filename)
        raise err


if __name__ == "__main__":
    pass
    # import pytest
    # pytest.main([__file__])
    # test_import_data()
