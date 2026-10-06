"""
Where pipeline artefacts live and what they are called.

The filename stems are defined once, here, and shared by the writer
(`PipelineConfig.get_model_path` / `get_data_path`) and every reader
(`latest_artifact`, the dashboard, serving). A reader and a writer that each
spelled the names out could disagree, and did.
"""

from __future__ import annotations

from pathlib import Path

#: Artefact filename stems, shared by the path builders on `PipelineConfig` and
#: by `latest_artifact` below. One definition, so a reader and a writer cannot
#: disagree about what a file is called - which is exactly how the dashboard
#: ended up looking for `Data_Prepared.csv` while the pipeline wrote
#: `Data_Prepared_<version>.csv`.
MODEL_STEMS: dict[str, str] = {
    "without_lag": "prediction_model_without_lag",
    "with_lag": "prediction_model_with_lag",
    "clustering": "pickup_cluster_model",
}

DATA_STEMS: dict[str, str] = {
    "clean": "clean_data",
    "prepared": "Data_Prepared",
    "with_lag": "data_with_lag",
    "without_lag": "data_without_lag",
}

def latest_artifact(output_dir: str | Path, data_type: str) -> Path | None:
    """
    Newest versioned dataset of a given type in `output_dir`.

    Every artefact is written as `<stem>_<model_version>.csv`, where the version
    is `%Y%m%d_%H%M%S`. That format sorts lexicographically in chronological
    order, so the newest run is simply the maximum name - no date parsing, and
    no dependence on filesystem timestamps, which copying a directory destroys.

    A consumer that hardcodes the unversioned name finds nothing after a
    successful run. The dashboard did exactly that, and rendered its empty state
    over a complete set of outputs.

    Args:
        output_dir: Directory the pipeline writes to.
        data_type: Key of `DATA_STEMS`, or a literal stem.

    Returns:
        Path to the newest match, the unversioned legacy file if that is all
        there is, or None when neither exists.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        return None

    stem = DATA_STEMS.get(data_type, data_type)
    # `.csv.gz` is what the pipeline writes now; bare `.csv` is what it wrote
    # before, and output directories from earlier runs should keep working.
    for pattern in (f"{stem}_*.csv.gz", f"{stem}_*.csv"):
        versioned = sorted(directory.glob(pattern))
        if versioned:
            return versioned[-1]

    for legacy in (directory / f"{stem}.csv.gz", directory / f"{stem}.csv"):
        if legacy.exists():
            return legacy
    return None


def latest_version(output_dir: str | Path) -> str | None:
    """
    Version string of the most recent run in `output_dir`.

    Read from the configuration snapshots, since those are written last and so
    only exist for runs that reached the end.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        return None
    snapshots = sorted(directory.glob("pipeline_config_*.json"))
    if not snapshots:
        return None
    return snapshots[-1].stem.removeprefix("pipeline_config_")
