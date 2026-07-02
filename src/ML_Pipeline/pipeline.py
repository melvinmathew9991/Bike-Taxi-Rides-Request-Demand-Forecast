"""
End-to-End ML Pipeline Orchestrator
Handles: Data Loading → Preprocessing → Feature Engineering → Model Training → Predictions
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from joblib import dump, load
from typing import Tuple, Dict, Any

from ML_Pipeline.data_prep_basic import data_prep_basic
from ML_Pipeline.data_prep_advanced import data_prep_advanced
from ML_Pipeline.data_prep_geospatial import data_prep_geospatial
from ML_Pipeline.model_training import model_training
from ML_Pipeline.prediction_pipeline import prediction_pipeline
from ML_Pipeline.utils import convert_into_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Comprehensive ML Pipeline for Bike-Taxi Demand Forecasting
    
    Pipeline Stages:
    1. Data Ingestion
    2. Data Cleaning & Basic Preprocessing
    3. Advanced Preprocessing & Feature Engineering
    4. Geospatial Clustering
    5. Model Training (with & without lag features)
    6. Model Evaluation
    7. Predictions
    """
    
    def __init__(self, 
                 raw_data_path: str,
                 output_dir: str = '../output',
                 test_data_path: str = None):
        """
        Initialize the ML Pipeline
        
        Args:
            raw_data_path: Path to raw data CSV/GZ
            output_dir: Directory to save models and processed data
            test_data_path: Path to test data (optional)
        """
        self.raw_data_path = raw_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_path = test_data_path
        
        self.df_raw = None
        self.df_processed = None
        self.models = {}
        self.metrics = {}
        
        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")
    
    def stage_1_load_data(self) -> pd.DataFrame:
        """Load raw data"""
        logger.info("=" * 60)
        logger.info("STAGE 1: Loading Raw Data")
        logger.info("=" * 60)
        
        try:
            # Try reading with gzip compression
            try:
                self.df_raw = pd.read_csv(
                    self.raw_data_path, 
                    low_memory=False, 
                    compression='gzip'
                )
            except (OSError, EOFError):
                # If gzip fails, try without compression
                self.df_raw = pd.read_csv(
                    self.raw_data_path, 
                    low_memory=False, 
                    compression=None
                )
            
            logger.info(f"Data loaded. Shape: {self.df_raw.shape}")
            logger.info(f"  Columns: {list(self.df_raw.columns)}")
            logger.info(f"  Memory usage: {self.df_raw.memory_usage().sum() / 1024**2:.2f} MB")
            return self.df_raw
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def stage_2_basic_preprocessing(self) -> pd.DataFrame:
        """Apply basic data cleaning and preprocessing"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Basic Preprocessing")
        logger.info("=" * 60)
        
        try:
            if self.df_raw is None:
                self.stage_1_load_data()
            
            start_time = datetime.now()
            self.df_processed = data_prep_basic(self.df_raw.copy())
            
            logger.info(f"[OK] Basic preprocessing completed")
            logger.info(f"  Shape after cleaning: {self.df_processed.shape}")
            logger.info(f"  Time taken: {datetime.now() - start_time}")
            
            # Save intermediate data
            self.df_processed.to_csv(
                self.output_dir / 'clean_data.csv', 
                index=False
            )
            logger.info(f"  Saved clean data to {self.output_dir / 'clean_data.csv'}")
            
            return self.df_processed
        
        except Exception as e:
            logger.error(f"[ERROR] Error in basic preprocessing: {str(e)}")
            raise
    
    def stage_3_advanced_preprocessing(self) -> pd.DataFrame:
        """Apply advanced data preparation"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Advanced Preprocessing")
        logger.info("=" * 60)
        
        try:
            if self.df_processed is None:
                self.stage_2_basic_preprocessing()
            
            start_time = datetime.now()
            clean_data_path = self.output_dir / 'clean_data.csv'
            
            self.df_processed = data_prep_advanced(
                self.df_processed.copy(),
                str(clean_data_path)
            )
            
            logger.info(f"[OK] Advanced preprocessing completed")
            logger.info(f"  Shape after advanced prep: {self.df_processed.shape}")
            logger.info(f"  Time taken: {datetime.now() - start_time}")
            
            return self.df_processed
        
        except Exception as e:
            logger.error(f"[ERROR] Error in advanced preprocessing: {str(e)}")
            raise
    
    def stage_4_geospatial_clustering(self) -> pd.DataFrame:
        """Apply geospatial clustering to create regions"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: Geospatial Clustering")
        logger.info("=" * 60)
        
        try:
            if self.df_processed is None:
                self.stage_3_advanced_preprocessing()
            
            start_time = datetime.now()
            cluster_model_path = str(self.output_dir / 'pickup_cluster_model.joblib')
            prepared_data_path = str(self.output_dir / 'Data_Prepared.csv')
            
            self.df_processed = data_prep_geospatial(
                self.df_processed.copy(),
                cluster_model_path,
                prepared_data_path
            )
            
            logger.info(f"[OK] Geospatial clustering completed")
            logger.info(f"  Number of clusters: {self.df_processed['pickup_cluster'].nunique()}")
            logger.info(f"  Time taken: {datetime.now() - start_time}")
            
            return self.df_processed
        
        except Exception as e:
            logger.error(f"[ERROR] Error in geospatial clustering: {str(e)}")
            raise
    
    def stage_5_model_training(self) -> Dict[str, Any]:
        """Train both lag and non-lag models"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5: Model Training")
        logger.info("=" * 60)
        
        try:
            if self.df_processed is None:
                self.stage_4_geospatial_clustering()
            
            model_without_lag_path = str(self.output_dir / 'prediction_model_without_lag.joblib')
            model_with_lag_path = str(self.output_dir / 'prediction_model_with_lag.joblib')
            
            logger.info("Training XGBoost models...")
            start_time = datetime.now()
            
            model_training(
                self.df_processed.copy(),
                model_without_lag_path,
                model_with_lag_path
            )
            
            # Load trained models
            self.models['without_lag'] = load(model_without_lag_path)
            self.models['with_lag'] = load(model_with_lag_path)
            
            logger.info(f"[OK] Model training completed")
            logger.info(f"  Models saved:")
            logger.info(f"    - {model_without_lag_path}")
            logger.info(f"    - {model_with_lag_path}")
            logger.info(f"  Total time taken: {datetime.now() - start_time}")
            
            return self.models
        
        except Exception as e:
            logger.error(f"[ERROR] Error in model training: {str(e)}")
            raise
    
    def stage_6_predictions(self) -> pd.DataFrame:
        """Generate predictions using trained models"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 6: Generating Predictions")
        logger.info("=" * 60)
        
        try:
            if not self.test_data_path:
                logger.warning("No test data path provided. Skipping predictions.")
                return None
            
            start_time = datetime.now()
            
            predictions = prediction_pipeline(
                cleaned_data_path=self.test_data_path,
                cluster_model_path=str(self.output_dir / 'pickup_cluster_model.joblib'),
                predict_without_lag_path=str(self.output_dir / 'prediction_model_without_lag.joblib'),
                predict_with_lag_path=str(self.output_dir / 'prediction_model_with_lag.joblib'),
                data_with_lag_path=str(self.output_dir / 'data_with_lag.csv'),
                data_without_lag_path=str(self.output_dir / 'data_without_lag.csv')
            )
            
            logger.info(f"[OK] Predictions generated")
            if predictions is not None:
                logger.info(f"  Predictions shape: {predictions.shape}")
            else:
                logger.info("  Predictions output not returned as a DataFrame")
            logger.info(f"  Time taken: {datetime.now() - start_time}")
            
            return predictions
        
        except Exception as e:
            logger.error(f"[ERROR] Error in prediction: {str(e)}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline from raw data to predictions"""
        logger.info("\n\n")
        logger.info("=" * 60)
        logger.info("BIKE-TAXI DEMAND FORECAST ML PIPELINE")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        pipeline_start = datetime.now()
        
        try:
            # Execute all pipeline stages
            self.stage_1_load_data()
            self.stage_2_basic_preprocessing()
            self.stage_3_advanced_preprocessing()
            self.stage_4_geospatial_clustering()
            self.stage_5_model_training()
            predictions = self.stage_6_predictions()
            
            total_time = datetime.now() - pipeline_start
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"[OK] Pipeline completed successfully!")
            logger.info(f"  Total execution time: {total_time}")
            logger.info(f"  Models trained: {len(self.models)}")
            logger.info(f"  Output directory: {self.output_dir}")
            logger.info("=" * 60)
            
            return {
                'status': 'success',
                'total_time': total_time,
                'models': self.models,
                'processed_data': self.df_processed,
                'predictions': predictions
            }
        
        except Exception as e:
            logger.error(f"\n[FAILED] Pipeline failed at {datetime.now()}")
            logger.error(f"  Error: {str(e)}")
            raise
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of each pipeline stage"""
        return {
            'data_loaded': self.df_raw is not None,
            'data_processed': self.df_processed is not None,
            'models_trained': len(self.models) > 0
        }


def main():
    """Example usage of the ML Pipeline"""
    
    # Initialize pipeline
    pipeline = MLPipeline(
        raw_data_path='../data/raw_data.csv',
        output_dir='../output',
        test_data_path='../data/test_dataset/cleaned_test_booking_data.csv'
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    # Access results
    print("\nPipeline Results:")
    print(f"Status: {results['status']}")
    print(f"Time: {results['total_time']}")
    print(f"Models: {list(results['models'].keys())}")


if __name__ == '__main__':
    main()
