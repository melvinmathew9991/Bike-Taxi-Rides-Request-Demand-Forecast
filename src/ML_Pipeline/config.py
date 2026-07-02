"""
Pipeline Configuration Manager
Centralized configuration for all pipeline settings and paths
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime


@dataclass
class PipelineConfig:
    """Configuration for ML Pipeline"""
    
    # Paths
    project_root: str = ''
    data_dir: str = 'data'
    raw_data_path: str = 'data/raw_data.csv'
    test_data_path: str = 'data/test_dataset/cleaned_test_booking_data.csv'
    output_dir: str = 'output'
    logs_dir: str = 'logs'
    
    # Model parameters
    test_size: float = 0.2
    train_day_cutoff: int = 23  # First 23 days for training
    test_day_cutoff: int = 24   # Last 7 days for testing
    
    # Clustering parameters
    n_clusters: int = 300  # Number of geographic clusters
    clustering_algorithm: str = 'kmeans'  # 'kmeans' or 'minibatch'
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Feature engineering
    lag_features: list = field(default_factory=lambda: [1, 2, 3])
    rolling_window: int = 3
    
    # Logging
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model registry
    save_models: bool = True
    save_intermediate_data: bool = True
    model_version: str = ''
    
    def __post_init__(self):
        """Initialize directories and set version"""
        if not self.model_version:
            self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directories
        for dir_path in [self.output_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_type: str) -> str:
        """Get path for a model"""
        models = {
            'without_lag': f'{self.output_dir}/prediction_model_without_lag_{self.model_version}.joblib',
            'with_lag': f'{self.output_dir}/prediction_model_with_lag_{self.model_version}.joblib',
            'clustering': f'{self.output_dir}/pickup_cluster_model_{self.model_version}.joblib',
        }
        return models.get(model_type, f'{self.output_dir}/{model_type}_{self.model_version}.joblib')
    
    def get_data_path(self, data_type: str) -> str:
        """Get path for processed data"""
        data_paths = {
            'clean': f'{self.output_dir}/clean_data_{self.model_version}.csv',
            'prepared': f'{self.output_dir}/Data_Prepared_{self.model_version}.csv',
            'with_lag': f'{self.output_dir}/data_with_lag_{self.model_version}.csv',
            'without_lag': f'{self.output_dir}/data_without_lag_{self.model_version}.csv',
        }
        return data_paths.get(data_type, f'{self.output_dir}/{data_type}_{self.model_version}.csv')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'raw_data_path': self.raw_data_path,
            'test_data_path': self.test_data_path,
            'output_dir': self.output_dir,
            'logs_dir': self.logs_dir,
            'test_size': self.test_size,
            'train_day_cutoff': self.train_day_cutoff,
            'test_day_cutoff': self.test_day_cutoff,
            'n_clusters': self.n_clusters,
            'clustering_algorithm': self.clustering_algorithm,
            'xgb_params': self.xgb_params,
            'lag_features': self.lag_features,
            'rolling_window': self.rolling_window,
            'model_version': self.model_version,
        }
    
    def save_config(self, filepath: str = None) -> str:
        """Save configuration to JSON file"""
        if filepath is None:
            filepath = f'{self.output_dir}/pipeline_config_{self.model_version}.json'
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        return filepath
    
    @staticmethod
    def load_config(filepath: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return PipelineConfig(**config_dict)


class ModelRegistry:
    """Registry to track all trained models and their metadata"""
    
    def __init__(self, registry_path: str = None):
        """Initialize model registry"""
        self.registry_path = registry_path or './model_registry.json'
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load existing registry or create new one"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def register_model(self, 
                      model_name: str,
                      model_path: str,
                      model_type: str,
                      metrics: Dict[str, float] = None,
                      parameters: Dict[str, Any] = None,
                      metadata: Dict[str, Any] = None) -> None:
        """Register a trained model"""
        
        self.registry[model_name] = {
            'model_path': model_path,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'parameters': parameters or {},
            'metadata': metadata or {},
        }
        
        self._save_registry()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a registered model"""
        return self.registry.get(model_name, None)
    
    def list_models(self, model_type: str = None) -> Dict[str, Dict[str, Any]]:
        """List all registered models, optionally filtered by type"""
        if model_type:
            return {k: v for k, v in self.registry.items() if v['model_type'] == model_type}
        return self.registry
    
    def get_best_model(self, model_type: str, metric: str = 'mse') -> Dict[str, Any]:
        """Get the best model by metric"""
        models_of_type = self.list_models(model_type)
        
        best_model = None
        best_value = float('inf')
        
        for name, info in models_of_type.items():
            if metric in info['metrics']:
                if info['metrics'][metric] < best_value:
                    best_value = info['metrics'][metric]
                    best_model = (name, info)
        
        return best_model
    
    def _save_registry(self) -> None:
        """Save registry to JSON file"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
    
    def export_registry(self, filepath: str) -> None:
        """Export registry to CSV for reporting"""
        import pandas as pd
        
        records = []
        for model_name, info in self.registry.items():
            record = {
                'model_name': model_name,
                'model_type': info['model_type'],
                'timestamp': info['timestamp'],
                **{f'metric_{k}': v for k, v in info.get('metrics', {}).items()},
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)


# Default configuration
DEFAULT_CONFIG = PipelineConfig()


if __name__ == '__main__':
    # Example usage
    config = PipelineConfig()
    print("Pipeline Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save configuration
    config_path = config.save_config()
    print(f"\nConfiguration saved to: {config_path}")
    
    # Model Registry example
    registry = ModelRegistry()
    print("\nModel Registry initialized")
