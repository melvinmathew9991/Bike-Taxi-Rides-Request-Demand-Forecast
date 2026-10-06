"""
Serving the promoted model.

    state.py        `ServingState`: the promoted model, its lag history and the
                    cluster centres, shared by the API, the monitor and the
                    dashboard so they cannot disagree about what is live
    api.py          the FastAPI app (`uvicorn ML_Pipeline.serving.api:app`)
    monitoring.py   scheduled health checks on the serving model

Importing this package does not import FastAPI; only `api` needs the
`serving` extra.
"""
