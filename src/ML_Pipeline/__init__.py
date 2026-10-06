"""
Bike-taxi ride-request demand forecasting.

Layout:
    config.py       PipelineConfig
    registry.py     ModelRegistry: metrics, gate verdicts, promotion
    artifacts.py    artefact file names, and finding the newest run
    governance.py   where personal data stops, as code
    pipeline.py     end-to-end orchestrator
    utils.py        CSV reading and distance helpers
    data/           raw bookings to the aggregated demand grid
    modeling/       features, training, forecasting, evaluation
    serving/        the promoted model, the API, health checks
    dashboard/      the Streamlit dashboard
    cli/            the `biketaxi` command
    features.py     compatibility only, for models saved before the split

Each subpackage's `__init__` lists its modules.
"""

__version__ = "1.0.0"
