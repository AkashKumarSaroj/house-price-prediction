"""
Week 3: Model Implementation and Cross-Validation Pipeline
Project: House Price Prediction (Ames Housing Dataset)
Author: Akash Kumar Saroj
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class HousingModelPipeline:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Ridge_Regression": Ridge(alpha=10.0, random_state=self.random_state),
            "Random_Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=self.random_state),
            "Gradient_Boosting": GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=self.random_state
            )
        }

    def validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("Features must be DataFrame, target must be Series/Array.")
        if X.isnull().sum().sum() > 0:
            raise ValueError("Input features contain unhandled NaNs.")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X ({len(X)}) vs y ({len(y)})")
        logging.info("Input validation passed.")

    def cross_validate_benchmarks(self, X: pd.DataFrame, y_log: pd.Series, n_splits: int = 5) -> pd.DataFrame:
        self.validate_inputs(X, y_log)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results = {}
        for name, model in self.models.items():
            neg_mse = cross_val_score(model, X, y_log, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
            rmse = np.sqrt(-neg_mse)
            results[name] = {
                "CV_RMSE_Mean": float(np.mean(rmse)),
                "CV_RMSE_Std": float(np.std(rmse))
            }
            logging.info(f"Model {name} - CV RMSE: {results[name]['CV_RMSE_Mean']:.4f}")
        return pd.DataFrame(results).T

    def train_and_evaluate_primary(self, X_train: pd.DataFrame, y_train_log: pd.Series,
                                   X_val: pd.DataFrame, y_val_log: pd.Series) -> Tuple[Any, Dict[str, float]]:
        best_model = self.models["Gradient_Boosting"]
        best_model.fit(X_train, y_train_log)
        preds_log = best_model.predict(X_val)
        
        y_val_true = np.expm1(y_val_log)
        preds_true = np.expm1(preds_log)
        
        metrics = {
            "RMSLE": float(np.sqrt(mean_squared_error(y_val_log, preds_log))),
            "MAE_USD": float(mean_absolute_error(y_val_true, preds_true)),
            "R2_Score": float(r2_score(y_val_true, preds_true))
        }
        return best_model, metrics

if __name__ == "__main__":
    print("Housing Model Pipeline Initialized Successfully.")
