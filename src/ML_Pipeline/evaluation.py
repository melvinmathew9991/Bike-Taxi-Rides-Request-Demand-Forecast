"""
Model Evaluation and Validation Module
Comprehensive evaluation metrics and model comparison
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            Dictionary containing all metrics
        """
        
        # Handle edge cases
        if len(y_true) == 0:
            logger.warning("Empty arrays provided to calculate_metrics")
            return {}
        
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Ensure same length
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        metrics = {}
        
        # Regression Metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        if not np.any(y_true == 0):  # Avoid division by zero
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        else:
            metrics['mape'] = np.nan
        
        # Additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Percentage metrics
        metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return metrics
    
    @staticmethod
    def compare_models(models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Compare multiple models' performance
        
        Args:
            models_dict: Dictionary of {model_name: (y_true, y_pred)}
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_name, (y_true, y_pred) in models_dict.items():
            metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
            metrics['model'] = model_name
            results.append(metrics)
        
        df_comparison = pd.DataFrame(results)
        return df_comparison.set_index('model')
    
    @staticmethod
    def get_best_model(comparison_df: pd.DataFrame, metric: str = 'rmse') -> str:
        """Get best model based on metric"""
        if metric == 'r2':
            return comparison_df[metric].idxmax()
        else:  # Lower is better for error metrics
            return comparison_df[metric].idxmin()
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual Plot"):
        """Plot residuals for diagnosis"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_predictions_over_time(y_true: np.ndarray, y_pred: np.ndarray, 
                                   time_index=None, title: str = "Predictions Over Time"):
        """Plot predictions vs actual over time"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if time_index is None:
            time_index = np.arange(len(y_true))
        
        ax.plot(time_index, y_true, label='Actual', linewidth=2, alpha=0.7)
        ax.plot(time_index, y_pred, label='Predicted', linewidth=2, alpha=0.7)
        ax.fill_between(time_index, y_true, y_pred, alpha=0.2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Request Count')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detailed error analysis"""
        errors = np.abs(y_true - y_pred)
        
        analysis = {
            'total_samples': len(y_true),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'percentile_25_error': np.percentile(errors, 25),
            'percentile_75_error': np.percentile(errors, 75),
            'percentile_90_error': np.percentile(errors, 90),
            'percentile_95_error': np.percentile(errors, 95),
        }
        
        # Error distribution
        analysis['errors_less_than_mean'] = np.sum(errors < np.mean(errors)) / len(errors) * 100
        analysis['errors_less_than_std'] = np.sum(errors < np.std(errors)) / len(errors) * 100
        
        return analysis
    
    @staticmethod
    def cross_validation_analysis(model, X, y, cv=5):
        """Perform cross-validation analysis"""
        from sklearn.model_selection import cross_val_score
        
        # MSE scores
        mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(mse_scores)
        
        # R2 scores
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        return {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores,
        }


class PredictionValidator:
    """Validate prediction quality and consistency"""
    
    @staticmethod
    def check_prediction_bounds(y_pred: np.ndarray, 
                                min_val: float = 0, 
                                max_val: float = None) -> Dict[str, Any]:
        """Check if predictions are within expected bounds"""
        
        out_of_bounds_low = np.sum(y_pred < min_val)
        out_of_bounds_high = np.sum(y_pred > max_val) if max_val else 0
        
        return {
            'out_of_bounds_low': out_of_bounds_low,
            'out_of_bounds_high': out_of_bounds_high,
            'percentage_out_of_bounds': (out_of_bounds_low + out_of_bounds_high) / len(y_pred) * 100,
            'min_prediction': np.min(y_pred),
            'max_prediction': np.max(y_pred),
            'mean_prediction': np.mean(y_pred),
        }
    
    @staticmethod
    def check_prediction_stability(y_pred: np.ndarray, window_size: int = 10) -> Dict[str, float]:
        """Check prediction stability across time windows"""
        
        stabilities = []
        for i in range(len(y_pred) - window_size):
            window = y_pred[i:i+window_size]
            stability = 1 - (np.std(window) / (np.mean(window) + 1e-8))
            stabilities.append(stability)
        
        return {
            'mean_stability': np.mean(stabilities),
            'min_stability': np.min(stabilities),
            'max_stability': np.max(stabilities),
            'std_stability': np.std(stabilities),
        }
    
    @staticmethod
    def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Comprehensive prediction validation"""
        
        validation = {
            'basic_metrics': ModelEvaluator.calculate_metrics(y_true, y_pred),
            'error_analysis': ModelEvaluator.error_analysis(y_true, y_pred),
            'bounds_check': PredictionValidator.check_prediction_bounds(y_pred),
            'stability_check': PredictionValidator.check_prediction_stability(y_pred),
        }
        
        return validation


def print_evaluation_report(metrics: Dict[str, float]):
    """Pretty print evaluation metrics"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name:.<40} {value:.4f}")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Example usage
    from numpy.random import RandomState
    
    rng = RandomState(42)
    y_true = rng.randint(0, 100, 100)
    y_pred = y_true + rng.normal(0, 10, 100)
    
    # Calculate metrics
    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
    print_evaluation_report(metrics)
    
    # Validation
    validator = PredictionValidator()
    validation = validator.validate_predictions(y_true, y_pred)
    print("Validation Results:")
    print(validation)
