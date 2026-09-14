"""
Week 4: Model Evaluation, Validation Protocols, and Residual Diagnostics
Project: House Price Prediction (Ames Housing Dataset)
Author: Akash Kumar Saroj
Role: ML Engineer Intern, YuvaIntern | BS CSDA, IIT Patna
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluationSuite:
    """
    Executes leak-free 5-fold cross validation, computes continuous
    regression metrics, and runs residual diagnostic distributions.
    """
    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.models = {
            "Ridge_Regression": Ridge(alpha=10.0, random_state=self.random_state),
            "Random_Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=self.random_state),
            "Gradient_Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=self.random_state)
        }

    def evaluate_cross_validation(self, X: pd.DataFrame, y_log: pd.Series) -> pd.DataFrame:
        """Calculates fold-by-fold RMSE and overall mean/std for each candidate model."""
        summary = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y_log, cv=self.kf, scoring="neg_mean_squared_error", n_jobs=-1)
            rmse_scores = np.sqrt(-scores)
            summary[name] = {
                "Mean_RMSLE": float(np.mean(rmse_scores)),
                "Std_RMSLE": float(np.std(rmse_scores)),
                "Min_RMSLE": float(np.min(rmse_scores)),
                "Max_RMSLE": float(np.max(rmse_scores))
            }
        return pd.DataFrame(summary).T

    def run_residual_diagnostics(self, model_name: str, X: pd.DataFrame, y_log: pd.Series):
        """Generates out-of-fold predictions, true dollar metrics, and statistical distribution of residuals."""
        model = self.models[model_name]
        oof_preds_log = cross_val_predict(model, X, y_log, cv=self.kf, n_jobs=-1)
        residuals = y_log - oof_preds_log

        # Back-transforming from log1p space to true currency values
        y_true = np.expm1(y_log)
        preds_true = np.expm1(oof_preds_log)

        diagnostics = {
            "RMSLE": float(np.sqrt(mean_squared_error(y_log, oof_preds_log))),
            "MAE_USD": float(mean_absolute_error(y_true, preds_true)),
            "R2_Score": float(r2_score(y_true, preds_true)),
            "Residual_Mean": float(np.mean(residuals)),
            "Residual_Std": float(np.std(residuals)),
            "Residual_Skewness": float(stats.skew(residuals)),
            "Residual_Kurtosis": float(stats.kurtosis(residuals))
        }

        # Failure Case Extraction
        error_df = pd.DataFrame({
            "True_SalePrice": y_true,
            "Predicted_Price": preds_true,
            "Absolute_Dollar_Error": np.abs(y_true - preds_true),
            "Percentage_Error": (np.abs(y_true - preds_true) / y_true) * 100
        })
        top_failures = error_df.sort_values(by="Absolute_Dollar_Error", ascending=False).head(5)

        return diagnostics, top_failures

if __name__ == "__main__":
    print("Model Evaluation Suite Initialized Successfully.")
