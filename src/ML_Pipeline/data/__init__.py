"""
From raw booking logs to the aggregated demand grid.

    prep_basic.py       deduplication, type coercion, per-rider gaps
    shift_time.py       the per-rider and per-pin gaps the cleaning rules read
    cleaning_rules.py   the business rules that drop rebookings, retries and
                        out-of-area pickups
    prep_advanced.py    the cleaning stage: applies the rules, writes clean data
    prep_geospatial.py  clusters pickups and aggregates to the demand grid
    clustering.py       offline cluster-count diagnostics

Booking-level personal data exists only in this subpackage. It stops at
`prep_geospatial`, whose output is counts per cluster per interval; see
`ML_Pipeline.governance` and docs/DATA_GOVERNANCE.md.
"""
