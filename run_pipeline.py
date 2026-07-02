"""
Main Pipeline Runner Script
Execute the complete ML pipeline with configuration and logging
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ML_Pipeline.pipeline import MLPipeline
from ML_Pipeline.config import PipelineConfig, ModelRegistry
from ML_Pipeline.evaluation import ModelEvaluator, print_evaluation_report


def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""
    
    if log_file is None:
        log_file = f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def run_pipeline(config: PipelineConfig = None, 
                 full_run: bool = True,
                 stages: list = None) -> dict:
    """
    Run the ML pipeline
    
    Args:
        config: Pipeline configuration
        full_run: Whether to run all stages
        stages: Specific stages to run (e.g., ['data', 'features', 'model', 'predict'])
    
    Returns:
        Dictionary with pipeline results
    """
    
    if config is None:
        config = PipelineConfig()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline Configuration:\n{json.dumps(config.to_dict(), indent=2)}")
    
    # Initialize pipeline
    pipeline = MLPipeline(
        raw_data_path=config.raw_data_path,
        output_dir=config.output_dir,
        test_data_path=config.test_data_path
    )
    
    results = {}
    
    try:
        if full_run or not stages:
            # Run complete pipeline
            logger.info("Running FULL pipeline...")
            results = pipeline.run_full_pipeline()
        else:
            # Run specific stages
            logger.info(f"Running stages: {stages}")
            
            if 'data' in stages:
                pipeline.stage_1_load_data()
                pipeline.stage_2_basic_preprocessing()
            
            if 'features' in stages:
                if pipeline.df_processed is None:
                    pipeline.stage_2_basic_preprocessing()
                pipeline.stage_3_advanced_preprocessing()
                pipeline.stage_4_geospatial_clustering()
            
            if 'model' in stages:
                if pipeline.df_processed is None:
                    pipeline.stage_4_geospatial_clustering()
                pipeline.stage_5_model_training()
            
            if 'predict' in stages:
                predictions = pipeline.stage_6_predictions()
                results['predictions'] = predictions
        
        # Save configuration
        config_path = config.save_config()
        logger.info(f"Configuration saved to: {config_path}")
        
        # Register models
        registry = ModelRegistry(f'{config.output_dir}/model_registry.json')
        
        for model_name, model in pipeline.models.items():
            registry.register_model(
                model_name=f"xgb_{model_name}_{config.model_version}",
                model_path=config.get_model_path(model_name),
                model_type='xgboost',
                parameters=config.xgb_params,
                metadata={'version': config.model_version, 'timestamp': datetime.now().isoformat()}
            )
        
        logger.info("Pipeline execution completed successfully.")
        logger.info(f"  Results saved to: {config.output_dir}")
        logger.info(f"  Model registry: {registry.registry_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='ML Pipeline Runner for Bike-Taxi Demand Forecast'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file',
        default=None
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        help='Specific stages to run (data, features, model, predict)',
        choices=['data', 'features', 'model', 'predict'],
        default=None
    )
    
    parser.add_argument(
        '--raw-data',
        type=str,
        help='Path to raw data file',
        default='data/raw_data.csv'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data file',
        default='data/test_dataset/cleaned_test_booking_data.csv'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for models and results',
        default='output'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        help='Number of geographic clusters',
        default=300
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path',
        default=None
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = PipelineConfig.load_config(args.config)
    else:
        config = PipelineConfig(
            raw_data_path=args.raw_data,
            test_data_path=args.test_data,
            output_dir=args.output,
            n_clusters=args.n_clusters
        )
    
    # Run pipeline
    try:
        results = run_pipeline(
            config=config,
            full_run=(args.stages is None),
            stages=args.stages
        )
        
        logger.info("Pipeline completed successfully.")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
