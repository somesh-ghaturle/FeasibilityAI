from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

class BaselineEstimator:
    def __init__(self, dataset: pd.DataFrame, target_column: str, task_type: str):
        self.df = dataset
        self.target_col = target_column
        self.task_type = task_type

    def get_baseline_performance(self):
        """
        Returns a baseline score (accuracy for classification, R2/neg_mse for regression).
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Simple preprocessing (drop NaN for baseline calculation)
        # In a real system, we'd handle this better, but for baseline, we want 'dumb' performance
        # We'll just drop rows for the estimator to run
        clean_df = self.df.dropna(subset=[self.target_col])
        if len(clean_df) == 0:
            return 0.0

        y = clean_df[self.target_col]
        X = clean_df.drop(columns=[self.target_col])
        
        # We only need 'y' for Dummy baseline usually, but sklearn API takes X
        
        try:
            if self.task_type == 'classification':
                # Majority class baseline
                model = DummyClassifier(strategy="most_frequent")
                scores = cross_val_score(model, X, y, cv=5)
                return scores.mean()
            elif self.task_type == 'regression':
                # Mean baseline
                model = DummyRegressor(strategy="mean")
                scores = cross_val_score(model, X, y, cv=5) # Default score is R2
                return scores.mean()
            else:
                return 0.0
        except ValueError:
            # E.g. too few samples for 5-fold
            return 0.0
