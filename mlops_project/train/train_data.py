import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from preprocess.preprocess_data import (DropMissing, IQR_DropOutliers,
                                        MissingIndicator, Standard_Scaler)
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class HousepricingDataPipeline:
    """
    A class representing the House-price data processing and modeling pipeline.

    Attributes:
        features (list): A list of numerical variables in the dataset.
        target (str): A string with the target feature name.
        n (int): Number of outliers permited on features.
        seed_split (int): Seed value to reproduce results.

    Methods:
        create_pipeline(): Creates and returns the House-pricing data processing pipeline.
    """

    def __init__(self, features, target, n, seed_model):
        self.FEATURES = features
        self.TARGET = target
        self.n = n
        self.SEED_MODEL = seed_model

    def create_pipeline(self):
        """
        Creates and returns the House-pricing data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE = Pipeline(
            [
                ('missing_indicator', MissingIndicator(variables=self.FEATURES)),
                ('iqr_dropoutliers', IQR_DropOutliers(features=self.FEATURES, n=self.n)),
                ('drop_missing', DropMissing()),
                # ('oh_encoder', OneHotEncoder(features=CATEGORICAL_FEATURES))
                ('scaler', Standard_Scaler(features=self.FEATURES, target=self.TARGET))
                # ('scaler', StandardScaler())
            ]
        )
        return self.PIPELINE

    def fit_random_forest(self, X_train, y_train):
        """
        Fit a Random Forest model using the predefined data preprocessing pipeline.

        Parameters:
            X_train (pandas.DataFrame or numpy.ndarray): The training input data.
            y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
            random_forest (RandomForest model): The fitted Rangom Forest model.
        """
        random_forest = RandomForestRegressor(n_estimators=100, random_state=self.SEED_MODEL)
        random_forest.fit(X_train, y_train)
        return random_forest

    def get_evaluation_metrics(self, model, X_train, y_train, X_test, y_test, y_pred):
        cv_score = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

        # Calculating Adjusted R-squared
        r2 = model.score(X_test, y_test)
        # Number of observations is the shape along axis 0
        n = X_test.shape[0]
        # Number of features (predictors, p) is the shape along axis 1
        p = X_test.shape[1]
        # Adjusted R-squared formula
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        R2 = model.score(X_test, y_test)
        CV_R2 = cv_score.mean()

        scores_df = pd.DataFrame(data=[[R2, adjusted_r2, CV_R2, RMSE]], columns=['R2 Score', 'Adjusted R2 Score', 'Cross Validated R2 Score', 'RMSE'])
        scores_df.insert(0, 'Model', 'Random Forest')

        return scores_df

    def persist_model(self, model, trained_model_dir, file_save_name):
        # Saves the model recently trained

        if not os.path.isdir(trained_model_dir):   # Searches for the default models folder
            os.mkdir(Path(trained_model_dir))
        if os.path.isdir(trained_model_dir):
            joblib.dump(model, trained_model_dir + file_save_name)

        print()
        print("Model stored in: " + trained_model_dir + file_save_name)
