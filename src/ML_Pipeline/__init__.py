"""
Bike-taxi ride-request demand forecasting.

Layout:
    features.py            canonical feature engineering (the single definition
                           shared by training and serving)
    splitting.py           chronological train/test splitting
    forecast.py            direct and recursive multi-step forecasting
    config.py              PipelineConfig
    registry.py            ModelRegistry: metrics, gate verdicts, promotion
    artifacts.py           artefact file names, and finding the newest run
    pipeline.py            end-to-end orchestrator
    data_prep_*.py         cleaning and aggregation stages
    model_training.py      training stage
    prediction_pipeline.py serving stage
    evaluation.py          metrics and prediction validation
    clustering.py          offline cluster-count diagnostics
    cli/                   the `biketaxi` command: run, registry, monitor, stage-demo
"""

__version__ = "1.0.0"
