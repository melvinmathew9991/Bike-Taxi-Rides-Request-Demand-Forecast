# Data governance

This project processes ride-booking records that constitute **personal data**.
This document records what that data is, why each field is retained, where the
personal data stops, and what the handling obligations are.

## 1. Data inventory

Source: booking-request logs, Bangalore, 2020-03-26 to 2021-03-26,
~8.38 M rows.

| Field | Type | Personal data? | Why retained |
|---|---|---|---|
| `ts` | timestamp | Indirect | The target variable is demand per interval. |
| `number` | integer | **Yes — pseudonymous identifier** | Needed to detect rebooking/retries by the same rider. Dropped at aggregation. |
| `pick_lat`, `pick_lng` | float (~6 dp) | **Yes — precise geolocation** | Assigns a booking to a geographic cluster. |
| `drop_lat`, `drop_lng` | float (~6 dp) | **Yes — precise geolocation** | Used only for the trip-distance sanity filter. |

### Why this is sensitive

- `number` is a **pseudonymous** identifier, not anonymous. It is stable across
  the year, so all of a rider's trips are linkable to one another. Under GDPR
  Recital 26 and India's DPDP Act 2023, pseudonymised data remains personal data.
- Coordinates at six decimal places resolve to roughly **0.1 m**. A rider's
  repeated early-morning pickups identify their home; repeated weekday drops
  identify their workplace. Re-identification from a handful of location points
  is well established in the literature and needs no auxiliary dataset beyond a
  public map.
- The combination — a stable identifier plus a precise location trace over a
  year — is materially more sensitive than either field alone.

### Applicable regime

The data is Indian (Bangalore), so the **Digital Personal Data Protection Act
2023** governs. Its principles relevant here: purpose limitation, data
minimisation, storage limitation, and reasonable security safeguards.

## 2. Where personal data stops

The pipeline has a hard boundary at the aggregation step:

```
raw_data.csv            booking-level    PERSONAL DATA
  -> data_prep_basic         booking-level    PERSONAL DATA
  -> data_prep_advanced      booking-level    PERSONAL DATA  -> clean_data.csv
  -> data_prep_geospatial    AGGREGATED       ---- boundary ----
  -> Data_Prepared.csv       counts per cluster per 30 min    NOT personal data
  -> model training          aggregated only
  -> forecasts               aggregated only
```

After `data_prep_geospatial`, every row is a count of requests in a geographic
cluster during a half-hour interval. No identifier, no coordinate, no individual
trip survives. **The trained models are fitted only on post-boundary data**, so
they cannot memorise an individual's movements.

### Enforcement in code

- `data_prep_advanced.PERSONAL_DATA_COLUMNS` names the personal fields, warns on
  every write of booking-level data, and supports `drop_rider_id=True` to omit
  the identifier from the persisted file.
- `data_prep_advanced.CLEANED_COLUMNS` is an explicit allow-list, so a new
  upstream column cannot silently start being written to disk.
- `streamlit_app.assert_no_personal_data()` refuses to render any file
  containing `number`, `pick_lat`, `pick_lng`, `drop_lat` or `drop_lng`. The
  dashboard therefore cannot expose personal data even if pointed at a
  booking-level file by mistake.

## 3. Handling rules

**Never commit data.** `.gitignore` excludes `/data/`, `/output/`, `*.csv`,
`*.gz`, `*.joblib`. Verify with `git status` before every commit. If personal
data is ever committed, rewriting history is not optional.

**Minimise what is persisted.** Only `clean_data.csv` needs booking-level
detail, and only as an intermediate. Prefer `drop_rider_id=True` unless the
rider identifier is genuinely required, and delete the intermediate once
`Data_Prepared.csv` exists.

**Restrict access.** Booking-level files should be readable only by those
running the pipeline. The aggregated grid can be shared freely — that is the
point of the boundary.

**Storage limitation.** Booking-level data should carry a retention period and
be deleted at its end. The aggregated grid may be retained indefinitely.

**Never send this data to a third-party service** — including hosted notebooks,
model APIs, or file-sharing links — without a lawful basis and a data-processing
agreement.

## 4. Ethical considerations

Beyond legal compliance, a demand-forecast model has distributional effects.

**Feedback loops.** Forecasts drive rider allocation. If the model under-predicts
demand in an area, fewer riders are sent, so fewer requests are fulfilled — and
the *next* training round sees lower demand there and predicts lower still.
Because the target is fulfilled requests rather than latent demand, this is
self-reinforcing. Areas that are already under-served are the ones most exposed.

*Mitigation:* monitor forecast error by cluster over time, and treat a cluster
whose predicted demand is falling monotonically as a candidate feedback loop
rather than a genuine trend. Where possible, log unfulfilled requests and search
events, not just completed bookings, so the target approximates real demand.

**Geographic equity.** Cluster sizes are not uniform — K-Means on a dense city
puts small tight clusters downtown and large sparse ones at the periphery. The
same absolute error means something very different in each. Reporting only a
global RMSE hides systematically worse service at the edges.

*Mitigation:* report error per cluster, and weight peripheral clusters
explicitly if service equity is an objective.

**Automation bias.** The model explains under half the variance in the target
(see `MODEL_CARD.md`). It is decision *support*, not a decision maker, and
should not be used for anything consequential to an individual — such as
individual rider pay or penalties.

**Purpose limitation.** This data was collected to operate a ride service. The
cluster model and booking history could be repurposed to profile individuals or
infer sensitive attributes from destinations (clinics, places of worship,
political venues). That is outside the collection purpose and should not be done.

## 5. Reproducibility without the data

The repository ships no data file, by design. The test suite therefore
constructs synthetic data in-memory (`tests/conftest.py`) and never depends on
the real dataset, so a fresh clone can run `pytest` immediately. See
`docs/DATA_SCHEMA.md` for the expected input format.
