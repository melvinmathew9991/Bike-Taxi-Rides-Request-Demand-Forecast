# ML Pipeline Implementation Guide

## How to Build and Run Your ML Pipeline

### Step 1: Verify Directory Structure

Ensure your project has this structure:
```
Bike-Taxi-Rides-Request-Demand-Forecast/
├── src/
│   ├── ML_Pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline.py              ← NEW: Main orchestrator
│   │   ├── config.py                ← NEW: Configuration manager
│   │   ├── evaluation.py            ← NEW: Model evaluation
│   │   ├── (existing modules...)
│   ├── engine.py
├── data/
│   ├── raw_data.csv
│   ├── clean_data.csv
│   └── test_dataset/
├── output/
├── Notebook/
├── run_pipeline.py                  ← NEW: Main runner script
├── ML_PIPELINE_README.md            ← NEW: Comprehensive guide
└── README.md
```

### Step 2: Create Required __init__.py Files

```bash
# Create __init__.py files if they don't exist
touch src/__init__.py
touch src/ML_Pipeline/__init__.py
```

### Step 3: Install Required Dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib geopy gpxpy matplotlib seaborn scipy
```

### Step 4: Run the Pipeline

#### Option A: Full Pipeline Execution
```bash
cd c:\Sachin\project\Bike-Taxi-Rides-Request-Demand-Forecast
python run_pipeline.py
```

#### Option B: Run Specific Stages
```bash
# Only preprocessing
python run_pipeline.py --stages data features

# Only model training
python run_pipeline.py --stages model

# Complete pipeline
python run_pipeline.py --stages data features model predict
```

#### Option C: Custom Configuration
```bash
python run_pipeline.py \
  --raw-data data/raw_data.csv \
  --output output \
  --n-clusters 250 \
  --log-file logs/my_pipeline.log
```

### Step 5: Monitor Execution

Check logs in real-time:
```bash
# PowerShell
Get-Content logs/pipeline_*.log -Tail 20 -Wait

# Unix/Linux
tail -f logs/pipeline_*.log
```

### Step 6: Review Results

After execution, check:
```
✓ output/clean_data.csv                    - Cleaned data
✓ output/Data_Prepared.csv                - Prepared data with features
✓ output/pickup_cluster_model.joblib      - Clustering model
✓ output/prediction_model_without_lag.joblib  - Model 1
✓ output/prediction_model_with_lag.joblib    - Model 2
✓ output/data_with_lag.csv                - Predictions (lag model)
✓ output/data_without_lag.csv             - Predictions (no lag)
✓ output/pipeline_config_*.json           - Configuration used
✓ output/model_registry.json              - Model metadata
✓ logs/pipeline_*.log                     - Execution logs
```

---

## Pipeline Architecture

### Stage Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

INPUT: Raw Data (CSV/GZ)
   │
   ├─→ [Stage 1] Load Data
   │   └─→ Read CSV, validate schema, log statistics
   │
   ├─→ [Stage 2] Basic Preprocessing
   │   ├─→ Remove duplicates
   │   ├─→ Convert data types
   │   ├─→ Handle missing values
   │   ├─→ Add time features (hour, day, month, etc.)
   │   └─→ Output: clean_data.csv
   │
   ├─→ [Stage 3] Advanced Preprocessing
   │   ├─→ Aggregate by time/region
   │   ├─→ Handle outliers
   │   ├─→ Normalize values
   │   └─→ Output: Data_Prepared.csv
   │
   ├─→ [Stage 4] Geospatial Clustering
   │   ├─→ Cluster pickup locations (300 clusters)
   │   ├─→ Assign cluster IDs
   │   └─→ Output: pickup_cluster_model.joblib
   │
   ├─→ [Stage 5] Model Training
   │   ├─→ Split: first 23 days (train), last 7 days (test)
   │   ├─→ Train Model 1: XGBoost without lag features
   │   ├─→ Train Model 2: XGBoost with lag features
   │   └─→ Output: prediction_model_*.joblib
   │
   ├─→ [Stage 6] Predictions
   │   ├─→ Load test data
   │   ├─→ Generate predictions from both models
   │   └─→ Output: data_with_lag.csv, data_without_lag.csv
   │
   OUTPUT: Trained Models + Predictions
```

### Key Components

#### 1. **MLPipeline Class** (pipeline.py)
```python
class MLPipeline:
    def stage_1_load_data()              # Load raw data
    def stage_2_basic_preprocessing()    # Clean data
    def stage_3_advanced_preprocessing() # Engineer features
    def stage_4_geospatial_clustering()  # Geographic regions
    def stage_5_model_training()         # Train models
    def stage_6_predictions()            # Generate predictions
    def run_full_pipeline()              # Execute all stages
    def get_pipeline_status()            # Check execution status
```

#### 2. **PipelineConfig Class** (config.py)
```python
class PipelineConfig:
    def __init__()           # Initialize with default settings
    def get_model_path()     # Get path to trained models
    def get_data_path()      # Get path to processed data
    def to_dict()            # Convert to dictionary
    def save_config()        # Save configuration to JSON
    def load_config()        # Load configuration from JSON
```

#### 3. **ModelRegistry Class** (config.py)
```python
class ModelRegistry:
    def register_model()     # Register a trained model
    def get_model_info()     # Get model metadata
    def list_models()        # List all registered models
    def get_best_model()     # Find best model by metric
    def export_registry()    # Export to CSV
```

#### 4. **ModelEvaluator Class** (evaluation.py)
```python
class ModelEvaluator:
    def calculate_metrics()      # Compute evaluation metrics
    def compare_models()         # Compare multiple models
    def get_best_model()         # Select best performer
    def plot_residuals()         # Visualize residuals
    def plot_predictions_over_time()  # Time series plot
    def error_analysis()         # Detailed error breakdown
```

---

## Usage Examples

### Example 1: Run Complete Pipeline (Simplest)
```python
# PowerShell or Command Prompt
cd C:\Sachin\project\Bike-Taxi-Rides-Request-Demand-Forecast
python run_pipeline.py
```

### Example 2: Run from Python Script
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from ML_Pipeline.pipeline import MLPipeline
from ML_Pipeline.config import PipelineConfig

# Create configuration
config = PipelineConfig(
    raw_data_path='data/raw_data.csv',
    output_dir='output',
    test_data_path='data/test_dataset/cleaned_test_booking_data.csv'
)

# Create and run pipeline
pipeline = MLPipeline(
    raw_data_path=config.raw_data_path,
    output_dir=config.output_dir,
    test_data_path=config.test_data_path
)

# Execute
results = pipeline.run_full_pipeline()

print(f"Status: {results['status']}")
print(f"Time: {results['total_time']}")
print(f"Models: {list(results['models'].keys())}")
```

### Example 3: Run Specific Stages
```python
pipeline = MLPipeline(...)

# Stage by stage execution
pipeline.stage_1_load_data()
print("✓ Data loaded")

pipeline.stage_2_basic_preprocessing()
print("✓ Basic preprocessing done")

pipeline.stage_3_advanced_preprocessing()
print("✓ Advanced preprocessing done")

pipeline.stage_4_geospatial_clustering()
print("✓ Clustering done")

pipeline.stage_5_model_training()
print("✓ Models trained")

predictions = pipeline.stage_6_predictions()
print("✓ Predictions generated")
```

### Example 4: Model Evaluation
```python
from ML_Pipeline.evaluation import ModelEvaluator, print_evaluation_report
import pandas as pd

# Load predictions
pred_data = pd.read_csv('output/data_with_lag.csv')
y_true = pred_data['actual']
y_pred = pred_data['predicted']

# Calculate metrics
metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
print_evaluation_report(metrics)

# Error analysis
errors = ModelEvaluator.error_analysis(y_true, y_pred)
print(f"Mean Error: {errors['mean_error']:.2f}")
print(f"Max Error: {errors['max_error']:.2f}")
print(f"95th Percentile Error: {errors['percentile_95_error']:.2f}")
```

### Example 5: Model Registry
```python
from ML_Pipeline.config import ModelRegistry

# Create registry
registry = ModelRegistry('output/model_registry.json')

# Get all XGBoost models
xgb_models = registry.list_models('xgboost')
print(f"Registered XGBoost models: {len(xgb_models)}")

# Find best model
best = registry.get_best_model('xgboost', metric='rmse')
print(f"Best model: {best[0]} with RMSE={best[1]['metrics']['rmse']:.4f}")

# Export for reporting
registry.export_registry('model_report.csv')
```

---

## Configuration Reference

### Default Configuration Values
```python
# Paths
raw_data_path: str = '../data/raw_data.csv'
output_dir: str = '../output'
test_data_path: str = '../data/test_dataset/cleaned_test_booking_data.csv'

# Data split
train_day_cutoff: int = 23  # First 23 days for training
test_day_cutoff: int = 24   # Last 7 days for testing

# Clustering
n_clusters: int = 300
clustering_algorithm: str = 'kmeans'

# XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'n_estimators': 100,
}

# Features
lag_features: list = [1, 2, 3]
rolling_window: int = 3
```

### Customize Configuration
```bash
# Via command line
python run_pipeline.py --n-clusters 500 --raw-data custom_data.csv

# Via JSON config file
python run_pipeline.py --config my_config.json
```

---

## Troubleshooting

### Problem: ModuleNotFoundError
```
Error: No module named 'ML_Pipeline'
```
**Solution**:
```bash
# Make sure you're in correct directory
cd c:\Sachin\project\Bike-Taxi-Rides-Request-Demand-Forecast

# Or set PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%\src
```

### Problem: FileNotFoundError
```
Error: [Errno 2] No such file or directory: 'data/raw_data.csv'
```
**Solution**:
```bash
# Check file exists and path is correct
python -c "import os; print(os.path.exists('data/raw_data.csv'))"

# Use absolute path if needed
python run_pipeline.py --raw-data C:\Sachin\project\...\raw_data.csv
```

### Problem: Out of Memory
```
Error: Unable to allocate memory
```
**Solution**:
```bash
# Reduce number of clusters
python run_pipeline.py --n-clusters 100

# Or run individual stages
python run_pipeline.py --stages data features
```

### Problem: Slow Execution
**Solutions**:
1. Reduce n_clusters: `--n-clusters 200`
2. Reduce n_estimators in config: `xgb_params['n_estimators'] = 50`
3. Use fewer lag features: `lag_features = [1]`
4. Check available system resources: `Task Manager` → `Performance`

---

## Performance Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MSE** | Mean((y - ŷ)²) | Average squared error (lower is better) |
| **RMSE** | √MSE | Same units as y, easier interpretation |
| **MAE** | Mean(\|y - ŷ\|) | Average absolute error |
| **R²** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1, higher is better) |
| **MAPE** | Mean(\|y - ŷ\|/y) * 100 | Percentage error |

---

## Next Steps

1. **Run the pipeline**: `python run_pipeline.py`
2. **Review outputs**: Check `output/` directory
3. **Evaluate models**: Use evaluation module to compare
4. **Integrate**: Use models in Streamlit app
5. **Deploy**: Package for production

---

## Useful Commands

```bash
# View latest logs
Get-Content logs/pipeline_*.log -Tail 50

# Check output files
Get-ChildItem output/ -Include *.csv, *.joblib, *.json

# Run with verbose logging
python run_pipeline.py --log-file logs/debug.log

# Load and inspect a model
python -c "from joblib import load; m = load('output/prediction_model_with_lag.joblib'); print(type(m))"

# Check data shape
python -c "import pandas as pd; df = pd.read_csv('output/clean_data.csv'); print(f'Shape: {df.shape}')"
```

---

## Support Files

- **Main Pipeline**: `src/ML_Pipeline/pipeline.py`
- **Configuration**: `src/ML_Pipeline/config.py`
- **Evaluation**: `src/ML_Pipeline/evaluation.py`
- **Runner Script**: `run_pipeline.py`
- **This Guide**: `ML_PIPELINE_IMPLEMENTATION.md`
- **Complete Guide**: `ML_PIPELINE_README.md`

Good luck! 🚀
