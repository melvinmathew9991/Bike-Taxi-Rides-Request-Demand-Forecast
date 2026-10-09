# Experiments

One-off measurements behind the figures in [docs/MODEL_CARD.md](../docs/MODEL_CARD.md).
They are research, not part of the pipeline or the service: nothing in `src/`
imports them, and the Docker image leaves them out. Operational scripts live in
[`scripts/`](../scripts).

| Script | Measures | Model card section |
|---|---|---|
| `compare_strategies.py` | four modelling strategies across five rolling origins, one step ahead and recursively | Rolling-origin validation |
| `measure_staleness.py` | how a frozen model decays, by weeks since training, from three origins | Model staleness |
| `measure_peak_error.py` | a stale vs a weekly-refitted model at the busiest cluster's peak hours | The promoted model was stale on the day it was trained |
| `fetch_weather.py` | downloads hourly Bengaluru weather from Open-Meteo into `data/` | — |
| `measure_weather.py` | whether weather or holidays improve the forecast (they do not) | Known limitations, item 2 |
| `measure_intervals.py` | coverage of Poisson against calibrated prediction intervals, on backtests the calibration did not use | Prediction intervals |

`common.py` holds the settings the sweep, peak-error and weather experiments
share. Everything in it is read from `PipelineConfig` except the fixed tree count
(`SWEEP_N_ESTIMATORS`), so the experiments measure the model the pipeline trains.

Run them from the repository root, against a real pipeline run:

```bash
python experiments/compare_strategies.py \
    --data output/Data_Prepared_<ver>.csv.gz \
    --cluster-model output/pickup_cluster_model_<ver>.joblib
```
