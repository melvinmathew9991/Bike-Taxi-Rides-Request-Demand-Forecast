"""
The Streamlit dashboard. Run it with `streamlit run streamlit_app.py`.

    app.py          layout, data source and navigation
    common.py       loading, validation and chart styling shared by every page
    overview.py     dataset size, span, schema and the target's distribution
    performance.py  whether the promoted model is fit to serve
    quality.py      completeness and time-grid integrity
    patterns.py     demand by hour, day of week and over time
    clusters.py     demand across geographic clusters
    forecasts.py    the forecast files the last run wrote

Every figure is computed from the aggregated demand grid the pipeline produces.
Nothing is hardcoded or sampled: if the data is not there, the page says so and
renders nothing. The grid carries no personal data, and
`ML_Pipeline.governance.assert_no_personal_data` refuses any file that does.

Needs the `dashboard` extra (streamlit, matplotlib).
"""
