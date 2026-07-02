# ML Pipeline Guide - Bike-Taxi Demand Forecast

## Overview

This is a **production-ready ML pipeline** for demand forecasting that handles:
- Data ingestion and validation
- Data preprocessing and cleaning
- Feature engineering
- Geospatial clustering
- Model training (XGBoost with/without lag features)
- Prediction and evaluation

## Architecture

```
Raw Data
    ↓
Stage 1: Data Loading
    ↓
Stage 2: Basic Preprocessing (cleaning, deduplication, type conversion)
    ↓
Stage 3: Advanced Preprocessing (aggregation, feature engineering)
    ↓
Stage 4: Geospatial Clustering (geographic region segmentation)
    ↓
Stage 5: Model Training (XGBoost - with lag & without lag variants)
    ↓
Stage 6: Predictions (inference on test data)
```

## Pipeline Components

### 1. **pipeline.py** - Main Orchestrator
Coordinates all pipeline stages with comprehensive logging.

**Key Class**: `MLPipeline`
- `stage_1_load_data()` - Load raw data
- `stage_2_basic_preprocessing()` - Clean and prepare data
- `stage_3_advanced_preprocessing()` - Feature engineering
- `stage_4_geospatial_clustering()` - Geographic clustering
- `stage_5_model_training()` - Train models
- `stage_6_predictions()` - Generate predictions
- `run_full_pipeline()` - Execute complete pipeline

### 2. **config.py** - Configuration Management
Centralized configuration and model registry.

**Key Classes**:
- `PipelineConfig` - Manages all pipeline settings and paths
- `ModelRegistry` - Tracks trained models and their metadata

### 3. **evaluation.py** - Model Evaluation
Comprehensive evaluation metrics and validation.

**Key Classes**:
- `ModelEvaluator` - Calculate metrics, compare models
- `PredictionValidator` - Validate predictions and check bounds

### 4. **run_pipeline.py** - Entry Point
Main script to run the pipeline from command line.

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```bash
# From project root directory
python run_pipeline.py
```

### Option 2: Run with Custom Configuration

```bash
python run_pipeline.py \
  --raw-data data/raw_data.csv \
  --test-data data/test_dataset/cleaned_test_booking_data.csv \
  --output output \
  --n-clusters 300
```

### Option 3: Run Specific Stages

```bash
# Run only data and feature engineering stages
python run_pipeline.py --stages data features

# Run only model training
python run_pipeline.py --stages model

# Run predictions only
python run_pipeline.py --stages predict
```

### Option 4: Use in Python Code

```python
from src.ML_Pipeline.pipeline import MLPipeline
from src.ML_Pipeline.config import PipelineConfig

# Initialize with custom configuration
config = PipelineConfig(
    raw_data_path='data/raw_data.csv',
    output_dir='output',
    n_clusters=300
)

# Create pipeline
pipeline = MLPipeline(
    raw_data_path=config.raw_data_path,
    output_dir=config.output_dir,
    test_data_path=config.test_data_path
)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Or run individual stages
pipeline.stage_1_load_data()
pipeline.stage_2_basic_preprocessing()
pipeline.stage_3_advanced_preprocessing()
pipeline.stage_4_geospatial_clustering()
pipeline.stage_5_model_training()
predictions = pipeline.stage_6_predictions()
```

## Configuration

### Default Configuration

Edit `src/ML_Pipeline/config.py` to customize:

```python
# Key parameters
n_clusters: int = 300  # Number of geographic regions
test_size: float = 0.2  # Test/train split
lag_features: list = [1, 2, 3]  # Lag features
rolling_window: int = 3  # Rolling window size

# XGBoost parameters
xgb_params = {
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 100,
    ...
}
```

### Custom Configuration File

1. Create a JSON config file:
```json
{
  "raw_data_path": "data/raw_data.csv",
  "output_dir": "output",
  "n_clusters": 300,
  "xgb_params": {
    "max_depth": 8,
    "learning_rate": 0.05
  }
}
```

2. Run with custom config:
```bash
python run_pipeline.py --config config.json
```

## Pipeline Stages Details

### Stage 1: Data Loading
- Loads raw CSV/GZ data
- Logs data shape and columns
- Validates data integrity

### Stage 2: Basic Preprocessing
- Remove duplicates
- Convert data types
- Add time features (hour, day, month, etc.)
- Handle missing values

**Output**: `clean_data.csv`

### Stage 3: Advanced Preprocessing
- Aggregate data by time and region
- Handle outliers
- Create temporal features
- Normalize values

**Output**: `Data_Prepared.csv`

### Stage 4: Geospatial Clustering
- Cluster pickup/dropoff locations
- Create geographic regions (default: 300 clusters)
- Calculate region centroids

**Output**: 
- `pickup_cluster_model.joblib` (KMeans model)
- Data with cluster assignments

### Stage 5: Model Training
- Prepare train/test splits (first 23 days = train, last 7 days = test)
- Train XGBoost WITHOUT lag features
- Train XGBoost WITH lag features (lag 1,2,3 + rolling mean)
- Evaluate both models

**Output**: 
- `prediction_model_without_lag.joblib`
- `prediction_model_with_lag.joblib`

### Stage 6: Predictions
- Load test data
- Apply clustering to test data
- Generate predictions using both models
- Save prediction results

**Output**: 
- `data_with_lag.csv`
- `data_without_lag.csv`

## Model Evaluation

### Metrics Calculated

- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R² Score** - Coefficient of Determination
- **MAPE** - Mean Absolute Percentage Error

### View Evaluation Results

```python
from src.ML_Pipeline.evaluation import ModelEvaluator, print_evaluation_report

# Calculate metrics
y_true = ...
y_pred = ...

metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
print_evaluation_report(metrics)

# Compare models
comparison = ModelEvaluator.compare_models({
    'model_1': (y_true_1, y_pred_1),
    'model_2': (y_true_2, y_pred_2),
})

best_model = ModelEvaluator.get_best_model(comparison, metric='rmse')
```

## Model Registry

Track all trained models and their metadata:

```python
from src.ML_Pipeline.config import ModelRegistry

# Initialize registry
registry = ModelRegistry('output/model_registry.json')

# Register a model
registry.register_model(
    model_name='xgb_with_lag_v1',
    model_path='output/prediction_model_with_lag.joblib',
    model_type='xgboost',
    metrics={'rmse': 5.2, 'r2': 0.85},
    parameters={'max_depth': 7, 'learning_rate': 0.1}
)

# Get model info
info = registry.get_model_info('xgb_with_lag_v1')

# Get best model by metric
best = registry.get_best_model('xgboost', metric='rmse')

# List all models
all_models = registry.list_models()

# Export registry to CSV
registry.export_registry('model_registry.csv')
```

## Output Files

After running the pipeline, you'll have:

```
output/
├── clean_data.csv                          # Stage 2 output
├── Data_Prepared.csv                       # Stage 3 output
├── pickup_cluster_model.joblib             # Clustering model
├── prediction_model_without_lag.joblib     # Model 1
├── prediction_model_with_lag.joblib        # Model 2
├── data_with_lag.csv                       # Predictions with lag
├── data_without_lag.csv                    # Predictions without lag
├── pipeline_config_*.json                  # Configuration snapshot
├── model_registry.json                     # Model metadata
└── logs/
    └── pipeline_*.log                      # Execution logs
```

## Troubleshooting

### Issue: "Module not found" error

```bash
# Make sure you're in the project root directory
cd c:\Sachin\project\Bike-Taxi-Rides-Request-Demand-Forecast

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: Out of memory

Reduce `n_clusters`:
```bash
python run_pipeline.py --n-clusters 100
```

### Issue: Slow execution

1. Check available system memory
2. Reduce dataset size for testing
3. Use specific stages instead of full pipeline:
```bash
python run_pipeline.py --stages data features
```

## Performance Optimization

### For Large Datasets

```python
config = PipelineConfig(
    n_clusters=100,  # Reduce clusters
    xgb_params={'n_estimators': 50}  # Reduce trees
)
```

### For Production

```python
config = PipelineConfig(
    n_clusters=500,  # More precision
    xgb_params={
        'n_estimators': 200,
        'max_depth': 8,
        'subsample': 0.9
    }
)
```

## Monitoring Pipeline Execution

### Check logs in real-time

```bash
# Unix/Linux/Mac
tail -f logs/pipeline_*.log

# Windows PowerShell
Get-Content logs/pipeline_*.log -Wait
```

### Programmatic monitoring

```python
from src.ML_Pipeline.pipeline import MLPipeline

pipeline = MLPipeline(...)

# Check status
status = pipeline.get_pipeline_status()
print(f"Data loaded: {status['data_loaded']}")
print(f"Data processed: {status['data_processed']}")
print(f"Models trained: {status['models_trained']}")
```

## Advanced Usage

### Custom Feature Engineering

Modify `src/ML_Pipeline/data_prep_advanced.py` to add custom features.

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
}

# Implement in model_training.py
```

### Cross-Validation

```python
from src.ML_Pipeline.evaluation import ModelEvaluator

cv_results = ModelEvaluator.cross_validation_analysis(
    model, X, y, cv=5
)
```

## Best Practices

1. **Always backup raw data** before running pipeline
2. **Save configuration** at each run for reproducibility
3. **Monitor logs** for warnings or errors
4. **Validate predictions** before deployment
5. **Version your models** using timestamps
6. **Track model performance** in the registry
7. **Run pipeline periodically** to retrain on new data

## Next Steps

1. ✅ Run the pipeline with default settings
2. ✅ Review output files and logs
3. ✅ Evaluate model performance
4. ✅ Integrate with Streamlit app
5. ✅ Deploy predictions to production

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review configuration in `output/pipeline_config_*.json`
3. Validate input data format and paths
4. Check system resources (memory, disk space)

---

**Version**: 1.0  
**Last Updated**: 2024  
**Maintainer**: Data Science Team
