import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score


class ModelPredictor:
    """
    A class representing the House-price data processing and modeling pipeline.

    Attributes:
        model:
        X_train:
        y_train:
        X_test:
        y_test:
        y_pred:
        scores:
        trained_model_dir:
        file_save_name:
        scores_df:

    Methods:
        predict:
        get_evaluation_metrics:
        persits_model:
    """

    def __init__(self, model, X_train, y_train, X_test, y_test, trained_model_dir, file_save_name):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.trained_model_dir = trained_model_dir
        self.file_save_name = file_save_name

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def get_evaluation_metrics(self):
        cv_score = cross_val_score(estimator=self.model, X=self.X_train, y=self.y_train, cv=10)

        # Calculating Adjusted R-squared
        r2 = self.model.score(self.X_test, self.y_test)
        # Number of observations is the shape along axis 0
        n = self.X_test.shape[0]
        # Number of features (predictors, p) is the shape along axis 1
        p = self.X_test.shape[1]
        # Adjusted R-squared formula
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, self.y_pred))
        R2 = self.model.score(self.X_test, self.y_test)
        CV_R2 = cv_score.mean()

        self.scores_df = pd.DataFrame(data=[[R2, adjusted_r2, CV_R2, RMSE]], columns=['R2 Score', 'Adjusted R2 Score', 'Cross Validated R2 Score', 'RMSE'])
        self.scores_df.insert(0, 'Model', 'Random Forest')

        return self.scores_df

    def persist_model(self):
        # Saves the model recently trained

        if not os.path.isdir(self.trained_model_dir):   # Searches for the default models folder
            os.mkdir(Path(self.trained_model_dir))
        if os.path.isdir(self.trained_model_dir):
            joblib.dump(self.model, self.trained_model_dir + self.file_save_name)

        print()
        print("Model stored in: " + self.trained_model_dir + self.file_save_name)
