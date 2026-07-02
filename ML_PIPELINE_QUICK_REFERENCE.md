# ML Pipeline - Quick Reference & Summary

## 🎯 What Was Built

A **production-ready ML pipeline** with 6 stages that automates the complete machine learning workflow for Bike-Taxi demand forecasting.

## 📁 New Files Created

### Core Pipeline Files
| File | Purpose |
|------|---------|
| `src/ML_Pipeline/pipeline.py` | Main orchestrator - runs all 6 pipeline stages |
| `src/ML_Pipeline/config.py` | Configuration management & model registry |
| `src/ML_Pipeline/evaluation.py` | Model evaluation metrics & validation |
| `run_pipeline.py` | Command-line entry point to run pipeline |

### Documentation Files
| File | Purpose |
|------|---------|
| `ML_PIPELINE_README.md` | Comprehensive guide (70+ sections) |
| `ML_PIPELINE_IMPLEMENTATION.md` | Step-by-step implementation guide |
| `ML_PIPELINE_QUICK_REFERENCE.md` | This file - quick commands |

---

## 🚀 Quick Start Commands

### Run Full Pipeline (Recommended)
```bash
cd C:\Sachin\project\Bike-Taxi-Rides-Request-Demand-Forecast
python run_pipeline.py
```

### Run Specific Stages
```bash
# Only data preprocessing
python run_pipeline.py --stages data features

# Only model training
python run_pipeline.py --stages model

# Only predictions
python run_pipeline.py --stages predict
```

### Custom Configuration
```bash
python run_pipeline.py \
  --raw-data data/raw_data.csv \
  --test-data data/test_dataset/cleaned_test_booking_data.csv \
  --output output \
  --n-clusters 300
```

---

## 🔄 Pipeline Stages (In Order)

```
1. Load Data          → Read CSV, validate
2. Basic Prep         → Clean, deduplicate, convert types
3. Advanced Prep      → Feature engineering, aggregation
4. Clustering         → Geographic region segmentation
5. Model Training     → Train XGBoost (with & without lag)
6. Predictions        → Generate forecasts
```

---

## 📊 Output Files

After running pipeline, you get:

| File | Content | Size |
|------|---------|------|
| `clean_data.csv` | Cleaned data after stage 2 | MB |
| `Data_Prepared.csv` | Features engineered data | MB |
| `pickup_cluster_model.joblib` | KMeans clustering model | KB |
| `prediction_model_without_lag.joblib` | XGBoost model (v1) | MB |
| `prediction_model_with_lag.joblib` | XGBoost model (v2) | MB |
| `data_with_lag.csv` | Predictions from model v2 | MB |
| `data_without_lag.csv` | Predictions from model v1 | MB |
| `pipeline_config_*.json` | Configuration snapshot | KB |
| `model_registry.json` | Model metadata | KB |
| `logs/pipeline_*.log` | Execution logs | MB |

---

## 💻 Python Usage Examples

### Example 1: Simple Pipeline Execution
```python
from src.ML_Pipeline.pipeline import MLPipeline

pipeline = MLPipeline(
    raw_data_path='data/raw_data.csv',
    output_dir='output'
)
results = pipeline.run_full_pipeline()
```

### Example 2: Run Stages Individually
```python
pipeline.stage_1_load_data()
pipeline.stage_2_basic_preprocessing()
pipeline.stage_3_advanced_preprocessing()
pipeline.stage_4_geospatial_clustering()
pipeline.stage_5_model_training()
predictions = pipeline.stage_6_predictions()
```

### Example 3: Model Evaluation
```python
from src.ML_Pipeline.evaluation import ModelEvaluator

metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### Example 4: Model Registry
```python
from src.ML_Pipeline.config import ModelRegistry

registry = ModelRegistry('output/model_registry.json')
best_model = registry.get_best_model('xgboost', metric='rmse')
print(f"Best: {best_model[0]}")
```

---

## 🔧 Configuration Customization

### Via Command Line
```bash
python run_pipeline.py --n-clusters 500 --log-file my_log.log
```

### Via Python
```python
from src.ML_Pipeline.config import PipelineConfig

config = PipelineConfig(
    n_clusters=250,
    xgb_params={'max_depth': 8, 'learning_rate': 0.05}
)
```

### Via JSON File
```bash
# Create config.json with custom settings
python run_pipeline.py --config config.json
```

---

## 📈 Key Metrics

Pipeline calculates:
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error  
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of Determination
- **MAPE** - Mean Absolute Percentage Error

View results after training completes.

---

## ⚡ Pipeline Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_clusters` | 300 | 50-1000 | Geographic regions |
| `max_depth` | 7 | 3-15 | XGBoost tree depth |
| `learning_rate` | 0.1 | 0.01-1.0 | XGBoost learning rate |
| `n_estimators` | 100 | 50-500 | Number of trees |
| `lag_features` | [1,2,3] | - | Lag periods |
| `rolling_window` | 3 | 2-10 | Window size |

---

## 🐛 Troubleshooting

### Error: Module not found
```bash
cd C:\Sachin\project\Bike-Taxi-Rides-Request-Demand-Forecast
python run_pipeline.py
```

### Error: File not found
```bash
# Verify file exists
python -c "import os; print(os.path.exists('data/raw_data.csv'))"
```

### Error: Out of memory
```bash
python run_pipeline.py --n-clusters 100
```

### Check logs
```bash
Get-Content logs/pipeline_*.log -Tail 50
```

---

## 📚 Documentation Files

1. **ML_PIPELINE_README.md** - Comprehensive (70+ sections)
   - Full architecture overview
   - Detailed stage-by-stage breakdown
   - Advanced usage examples
   - Performance optimization

2. **ML_PIPELINE_IMPLEMENTATION.md** - Step-by-step guide
   - Implementation instructions
   - Verification checklist
   - Example scripts
   - Troubleshooting reference

3. **ML_PIPELINE_QUICK_REFERENCE.md** - This file
   - Quick commands
   - Key shortcuts
   - At-a-glance reference

---

## ✅ Verification Checklist

Before running:
- [ ] Python 3.7+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Raw data exists: `data/raw_data.csv`
- [ ] Output directory writable: `output/`
- [ ] Sufficient disk space (check logs/)

After running:
- [ ] Check logs for errors: `logs/pipeline_*.log`
- [ ] Verify output files created: `output/*.csv`, `output/*.joblib`
- [ ] Review model registry: `output/model_registry.json`
- [ ] Check model performance: metrics in logs

---

## 🎯 Next Steps

1. **Run pipeline**: 
   ```bash
   python run_pipeline.py
   ```

2. **Monitor execution**:
   ```bash
   Get-Content logs/pipeline_*.log -Wait
   ```

3. **Check results**:
   ```bash
   Get-ChildItem output/ | Select-Object Name, Length
   ```

4. **Review predictions**:
   ```python
   import pandas as pd
   df = pd.read_csv('output/data_with_lag.csv')
   print(df.head())
   ```

5. **Integrate with Streamlit**:
   - Load models from `output/`
   - Use predictions for display

---

## 📞 File Locations

| Component | Location |
|-----------|----------|
| Pipeline | `src/ML_Pipeline/pipeline.py` |
| Configuration | `src/ML_Pipeline/config.py` |
| Evaluation | `src/ML_Pipeline/evaluation.py` |
| Entry point | `run_pipeline.py` |
| Logs | `logs/pipeline_*.log` |
| Models | `output/*.joblib` |
| Data | `output/*.csv` |
| Metadata | `output/*.json` |

---

## 🚀 Success Indicators

✅ Pipeline is working correctly when you see:
- All 6 stages complete with "✓" marks
- No error messages in logs
- All output files created
- Metrics printed to console
- Model registry populated

---

## 💡 Pro Tips

1. **First run**: Start with default settings
2. **Debugging**: Check specific stages with `--stages data features`
3. **Performance**: Reduce `n_clusters` for faster runs
4. **Monitoring**: Use `tail -f logs/pipeline_*.log` for real-time updates
5. **Reproducibility**: Save config JSON for later reruns

---

## Resources

- Main Documentation: `ML_PIPELINE_README.md`
- Implementation Guide: `ML_PIPELINE_IMPLEMENTATION.md`
- Quick Reference: This file
- Source Code: `src/ML_Pipeline/`
- Entry Point: `run_pipeline.py`

---

**Ready to run?** Start with:
```bash
python run_pipeline.py
```

Good luck! 🚀
