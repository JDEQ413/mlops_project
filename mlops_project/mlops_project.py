"""Main module."""

from load.load_data import DataRetriever
from predictor.model_predictor import ModelPredictor
from sklearn.model_selection import train_test_split
from train.train_data import HousepricingDataPipeline

MAIN_DIR = './mlops_project/'
DATASETS_DIR = MAIN_DIR + 'data/'
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

TRAINED_MODEL_DIR = MAIN_DIR + 'models/'
PIPELINE_NAME = 'random_forest'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

droped_rows_index_list = []

if __name__ == "__main__":

    # print(os.getcwd())
    # os.chdir(FULL_USER_DIR)

    # Retrieve data
    data_retriever = DataRetriever([DATASETS_DIR, KAGGLE_URL, KAGGLE_LOCAL_DIR, DATA_RETRIEVED])
    result = data_retriever.retrieve_data()

    # Read data
    raw_df = data_retriever.load_data()
    print(raw_df)

    # House-pricing Pipeline
    housepricing_pipeline = HousepricingDataPipeline(features=FEATURES, target=TARGET, n=1, seed_model=SEED_MODEL)
    housepricing_pipeline.create_pipeline()
    df_transformed = housepricing_pipeline.PIPELINE.fit_transform(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop(TARGET, axis=1), df_transformed[TARGET], test_size=0.2, random_state=SEED_SPLIT)

    # Creating and training model
    RF_model = housepricing_pipeline.fit_random_forest(X_train=X_train, y_train=y_train)
    RF_model.fit(X_train, y_train)

    # Model making a prediction on test data, and persisting the model
    predictor = ModelPredictor(model=RF_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, trained_model_dir=TRAINED_MODEL_DIR, file_save_name=PIPELINE_SAVE_FILE)
    predictor.predict()
    print(predictor.get_evaluation_metrics())
    predictor.persist_model()

    # Predictions
