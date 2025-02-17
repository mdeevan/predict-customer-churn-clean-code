'''
identify credit card customers who are most likely to churn.
the testing script is to test the functionalities defined in churn_library.py
This version makes use of the pytest classes

Author: Muhammad Naveed
Created On : Feb 5th, 2025
'''

import os
import logging
import pytest

import constants
import churn_library_class as cls


class TestCustomerChurn():
    '''
    declare class level properties, as each test depends upon the results from
    the previous test
    '''
    datafilename = None
    df = None
    nrows = 50
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    X_data = None
    y_data = None
    y_train_preds_rf = None
    y_test_preds_rf = None
    y_train_preds_lr = None
    y_test_preds_lr = None
    model = None
    cc_obj = None
    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        force=True,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    def setup_class(self):
        '''
        create object necessary for the test
        '''
        # https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging

        TestCustomerChurn.datafilename = "bank_data.csv"
        TestCustomerChurn.cc_obj = cls.CustomerChurn(nrows=50)

        TestCustomerChurn.logging = logging.getLogger()

        TestCustomerChurn.logging.info("setup_class")
        TestCustomerChurn.logging.info("class created, df has %s rows",
                                       TestCustomerChurn.cc_obj.nrows)
        TestCustomerChurn.logging.info("class created, filename is %s ",
                                       TestCustomerChurn.cc_obj.filename)

    def teardown_class(self):
        '''
        release the object memory at the end of testing
        '''
        TestCustomerChurn.cc_obj = None
        TestCustomerChurn.logging = None

    def test_import_data(self):
        '''
        test data import - this example is completed for you to assist with the
        other test functions
        INPUT:
            none
        OUTPUT:
            none
        '''

        try:
            TestCustomerChurn.cc_obj._import_data()

            TestCustomerChurn.logging.info("Testing import_data: SUCCESS")

        except FileNotFoundError as err:
            TestCustomerChurn.logging.error("Testing import_eda: The file \
                wasn't found")
            raise err

        try:
            assert TestCustomerChurn.cc_obj.df.shape[0] > 0
            assert TestCustomerChurn.cc_obj.df.shape[1] > 0

        except AssertionError as err:
            TestCustomerChurn.logging.error("Testing import_data: The file \
            doesn't appear to have rows and columns")
            raise err

    def test_perform_eda(self, tmp_path):
        '''
        test perform eda function
        INPUT:
            tmp_path: Fixture: retrieve the temporary path for storing the 
            files created during the test execution
        OUTPUT:
            None
        '''
        try:
            TestCustomerChurn.cc_obj._perform_eda(tmp_path)
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

    def test_encoder_helper(self):
        '''
        test encoder helper
        INPUT:
            df: Fixture: to Retrieve the data to use for the test
        OUTPUT:
            data: dataframe : Returns the dataframe modified via
            encoder-helper procedure
        '''

        try:
            TestCustomerChurn.cc_obj._encoder_helper()
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
                if TestCustomerChurn.cc_obj.df[col_name].shape[0] > 0:
                    logging.info("%s successfully created : SUCCESS", col_name)

            except KeyError as err:
                logging.error("%s not created : FAILURE", col_name)
                raise err

    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''

        try:
            TestCustomerChurn.cc_obj._perform_feature_engineering()

            logging.info("perform feature engineering: SUCCESS")

        except (AssertionError, KeyError, ValueError) as err:
            logging.error("perform feature engineering: FAILURE")
            raise err

        lst_obj = [TestCustomerChurn.cc_obj.X_train,
                   TestCustomerChurn.cc_obj.X_test,
                   TestCustomerChurn.cc_obj.y_train,
                   TestCustomerChurn.cc_obj.y_test]
        lst_str = ["X_train", "X_test", "y_train", "y_test"]

        for i, obj in enumerate(lst_obj):
            try:
                if obj.shape[0] > 0:
                    logging.info("%s has a shape of {obj.shape}", lst_str[i])

            except (AssertionError, KeyError) as err:
                logging.error("%s failed to be created", lst_str[i])
                raise err

    def test_train_models(self, tmp_path):
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

        # data = cls.encoder_helper(df_in, constants.CAT_COLUMNS, constants.SUFFIX)
        # # X_data, y_data, X_train, X_test, y_train, y_test = cls.perform_feature_engineering(data)
        # _, _, split_data = cls.perform_feature_engineering(data)

        try:
            TestCustomerChurn.cc_obj._train_models(tmp_path, tmp_path)

            if TestCustomerChurn.cc_obj.model is not None:
                logging.info("Training model : SUCCESS")
            else:
                logging.info("Training model : FAILURE")

        except AssertionError as err:
            logging.error("Training model: FAILURE")
            raise err

    def test_feature_importance_plot(self, tmp_path):
        '''
        test feature importance plot
        INPUT:
            df: Fixture:  retrieve the dataframe from the Fixture defined earlier
            tmp_path: Fixture: retrieve the temporary path for storing the files
            created during the test execution
        OUTPUT:
            None

        '''
        try:
            TestCustomerChurn.cc_obj._feature_importance_plot(tmp_path)
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
