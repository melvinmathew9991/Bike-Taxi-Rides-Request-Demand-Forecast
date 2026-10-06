"""
Bike-Taxi Demand Forecast - analysis dashboard.

    streamlit run streamlit_app.py

The dashboard lives in `ML_Pipeline.dashboard`, one module per page. This file
stays at the repository root because it is what `streamlit run` is pointed at.
Set `BIKETAXI_OUTPUT_DIR` to read another run's output directory.
"""

from ML_Pipeline.dashboard.app import main

main()
