import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
# from xgboost import XGBClassifier, XGBRegressor # XGBoost removed due to missing libomp dependency
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

class MLEstimator:
    def __init__(self, dataset: pd.DataFrame, target_column: str, task_type: str):
        self.df = dataset
        self.target_col = target_column
        self.task_type = task_type

    def estimate_performance(self):
        """
        Trains quick ML models and returns the best CV score.
        Models:
        1. Linear/Logistic (Simple baseline)
        2. RandomForest (Robust bagging)
        3. GradientBoosting (Boosting)
        4. MLP (Neural Network / Deep Learning proxy)
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Pipelining
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        if self.task_type == 'classification':
            models = [
                ('LogisticRegression', LogisticRegression(max_iter=1000)),
                ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('GradientBoosting', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('NeuralNetwork (MLP)', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
            ]
        elif self.task_type == 'regression':
            models = [
                ('LinearRegression', LinearRegression()),
                ('RandomForest', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('GradientBoosting', GradientBoostingRegressor(n_estimators=50, random_state=42)),
                ('NeuralNetwork (MLP)', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
            ]
        else:
            return 0.0, 0.0, "None"

        best_score = -float('inf')
        best_std = 0.0
        best_model_name = "None"

        for name, model in models:
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            try:
                # 5-fold CV
                scores = cross_val_score(clf, X, y, cv=5)
                avg_score = scores.mean()
                if avg_score > best_score:
                    best_score = avg_score
                    best_std = scores.std()
                    best_model_name = name
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue

        return best_score, best_std, best_model_name
