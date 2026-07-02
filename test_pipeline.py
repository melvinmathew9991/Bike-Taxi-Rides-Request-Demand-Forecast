#!/usr/bin/env python
"""Quick test script for pipeline"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ML_Pipeline.pipeline import MLPipeline
from ML_Pipeline.config import PipelineConfig

print("Testing ML Pipeline...")
print("=" * 60)

# Create config
config = PipelineConfig(
    raw_data_path='data/raw_data.csv',
    test_data_path='data/test_dataset/cleaned_test_booking_data.csv',
    output_dir='output'
)

# Create pipeline
pipeline = MLPipeline(
    raw_data_path=config.raw_data_path,
    output_dir=config.output_dir,
    test_data_path=config.test_data_path
)

# Test Stage 1
print("\n[STAGE 1] Loading Data...")
try:
    pipeline.stage_1_load_data()
    print(f"[OK] Data loaded successfully")
    print(f"  Shape: {pipeline.df_raw.shape}")
    print(f"  Columns: {list(pipeline.df_raw.columns[:5])}...")
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

print("\nNow running full pipeline...")
print("=" * 60)

# Run full pipeline
try:
    results = pipeline.run_full_pipeline()
    print("\n[OK] Pipeline completed successfully!")
except Exception as e:
    print(f"\n[ERROR] Pipeline failed: {e}")
    sys.exit(1)
