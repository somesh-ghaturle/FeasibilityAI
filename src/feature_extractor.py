import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class FeatureExtractor:
    def __init__(self, dataset: pd.DataFrame, target_column: str, task_type: str):
        self.df = dataset
        self.target_col = target_column
        self.task_type = task_type

    def extract_features(self):
        """
        Extracts meta-features from the dataset.
        """
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset")

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # 1. Basic Stats
        n_samples = len(self.df)
        n_features = X.shape[1]
        
        # 2. Missing Values
        missing_ratio = self.df.isnull().mean().mean()
        
        # 3. Handle categorical target for metrics
        if self.task_type == 'classification':
             # Cardinality and Entropy of target
            y_counts = y.value_counts(normalize=True)
            label_entropy = entropy(y_counts)
            imbalance_ratio = y_counts.max() / y_counts.min() if len(y_counts) > 0 else 0
        else:
            # For regression, entropy/imbalance are different/less relevant in this simple form
            label_entropy = 0.0
            imbalance_ratio = 1.0 # Placeholder

        # 4. Feature Cardinality (Avg unique values per column for categorical-like cols)
        # We'll treat object/category columns as categorical
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            feature_cardinality_avg = X[cat_cols].nunique().mean()
        else:
            feature_cardinality_avg = 0.0

        # 5. Signal to Noise Estimation
        # Heuristic: Train a simple shallow tree. If it fails to find signal, data might be noise.
        signal_to_noise = self._estimate_signal_to_noise(X, y)

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "missing_ratio": missing_ratio,
            "label_entropy": label_entropy,
            "imbalance_ratio": imbalance_ratio,
            "feature_cardinality_avg": feature_cardinality_avg,
            "signal_to_noise_est": signal_to_noise
        }

    def _estimate_signal_to_noise(self, X, y):
        # Handle simple preprocessing for the proxy model
        # Fill missing numeric with 0, drop others for speed
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        
        # If no numeric features, maybe encode one or two? 
        # For speed/robustness of this specific 'signal' check, let's stick to numeric 
        # or simple encoding if empty.
        if X_num.shape[1] == 0:
            # Fallback for purely categorical: Label Encode everything just for a quick signal check
            X_num = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

        if len(y) < 20: 
            return 0.0 # Too few samples to trust CV

        try:
            if self.task_type == 'classification':
                # Shallow decision tree
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                # Quick 3-fold CV
                scores = cross_val_score(model, X_num, y, cv=3)
                return scores.mean()
            else:
                # Regression stub
                return 0.5 
        except Exception as e:
            # If model fails (e.g. too few classes), return 0
            print(f"Signal est failed: {e}")
            return 0.0
