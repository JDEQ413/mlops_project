from preprocess.preprocess_data import (DropMissing, IQR_DropOutliers,
                                        MissingIndicator, Standard_Scaler)
from sklearn.ensemble import RandomForestRegressor
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
