import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

class LightGBMModel:
    """
    LightGBM baseline model using tabular LOB features.
    Predicts the mid-price direction (Down, Flat, Up).
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None

    def prepare_data(self, df, label_col='label', drop_cols=None):
        """
        Splits features and label. Drops leakage columns like future prices.
        """
        if drop_cols is None:
            # Drop price levels directly and timestamp to prevent absolute price leakage.
            # We want the model to learn from imbalances, spreads, and relative metrics!
            # Also drop any timestamp or index columns.
            drop_cols = ['timestamp', 'last_update_id', 'label']
            # Drop raw price levels if they are in the dataframe, keeping volumes, spreads and imbalances
            price_cols = [c for c in df.columns if 'bid_p_' in c or 'ask_p_' in c or 'mid_price' in c or 'micro_price' in c or 'weighted_mid' in c]
            drop_cols.extend(price_cols)
            
        features = [c for c in df.columns if c not in drop_cols]
        X = df[features]
        y = df[label_col]
        
        # Filter out rows with sentinel labels (-1)
        valid_mask = y != -1
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM classifier.
        """
        print(f"Training LightGBM on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        # Initialize LightGBM Classifier
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model and prints performance metrics.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
            
        preds = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, labels=[0, 1, 2], target_names=['Down', 'Flat', 'Up'], output_dict=True)
        
        print("\n--- LightGBM Evaluation Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, labels=[0, 1, 2], target_names=['Down', 'Flat', 'Up']))
        
        return acc, report

    def save(self, filepath):
        """Saves the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Saved LightGBM model to {filepath}")

    def load(self, filepath):
        """Loads a model checkpoint from disk."""
        self.model = joblib.load(filepath)
        print(f"Loaded LightGBM model from {filepath}")
        
    def predict(self, X):
        """Predicts classes."""
        if self.model is None:
            raise ValueError("Model is not loaded or trained!")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts class probabilities."""
        if self.model is None:
            raise ValueError("Model is not loaded or trained!")
        return self.model.predict_proba(X)
