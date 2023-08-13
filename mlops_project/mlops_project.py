"""Main module."""

import os

from load.load_data import DataRetriever

DATASETS_DIR = './datasets/'
KAGGLE_URL = "https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data"
KAGGLE_LOCAL_DIR = KAGGLE_URL.split('/')[-1]
DATA_RETRIEVED = 'data.csv'
FULL_USER_DIR = "Users/usuario/Documents/GitHub/mlops_project/mlops_project/"

COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
TARGET = 'MEDV'
FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
NUMERICAL_FEATURES = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
CATEGORICAL_FEATURES = ['CHAS', 'RAD']
SELECTED_FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

SEED_SPLIT = 42
SEED_MODEL = 102

TRAINED_MODEL_DIR = 'trained_models/'
PIPELINE_NAME = 'random_forest'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

droped_rows_index_list = []

if __name__ == "__main__":

    print(os.getcwd())
    os.chdir(FULL_USER_DIR)

    # Retrieve data
    data_retriever = DataRetriever([DATASETS_DIR, KAGGLE_URL, KAGGLE_LOCAL_DIR, DATA_RETRIEVED])
    result = data_retriever.retrieve_data()
    print(result)
