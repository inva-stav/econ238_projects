#!/usr/bin/env python3
"""
Fetch 2024 ERCOT hourly data for Task 4 demand response model.

Outputs (saved to data/):
  prices_2024.csv   -- RTM settlement point prices, HB_BUSAVG hub, $/MWh
  solar_cf_2024.csv -- Hourly solar capacity factors (ERCOT system-wide, 0-1)
  wind_cf_2024.csv  -- Hourly wind capacity factors (ERCOT system-wide, 0-1)

All timestamps are UTC, hourly, covering Jan 1 – Dec 31 2024 (8784 hours).

Notes:
  - Solar and wind capacity factors are ERCOT system-wide (not zone-specific),
    normalized by each fuel's peak generation in 2024.
  - ERCOT solar is dominated by West Texas; wind by West Texas + Panhandle.
  - Prices use the HB_BUSAVG settlement point (bus-weighted hub average).
  - Source for solar/wind: ERCOT Fuel Mix Report (IntGenbyFuel2024.xlsx, 15-min)
  - Source for prices: gridstatus ERCOT RTM SPP (15-min, aggregated to hourly)
"""

import io
import os
import warnings
import zipfile

import gridstatus
import pandas as pd
import requests

warnings.filterwarnings("ignore")

FUEL_MIX_URL = (
    "https://www.ercot.com/files/docs/2021/03/10/FuelMixReport_PreviousYears.zip"
)
SETTLEMENT_POINT = "HB_BUSAVG"
MONTH_SHEETS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def fetch_rtm_prices() -> pd.Series:
    """Return 2024 hourly RTM prices for HB_BUSAVG in $/MWh, indexed by UTC datetime."""
    print("Fetching 2024 RTM settlement point prices via gridstatus...")
    ercot = gridstatus.Ercot()
    df = ercot.get_rtm_spp(year=2024)
    hub = df[df["Location"] == SETTLEMENT_POINT].copy()
    # Use interval start for left-aligned hourly buckets
    hub = hub.set_index("Interval Start")["SPP"]
    hub.index = pd.DatetimeIndex(hub.index).tz_convert("UTC")
    hourly = hub.resample("h").mean()
    hourly.index.name = "datetime_utc"
    hourly.name = "lmp_usd_mwh"
    return hourly


def _parse_fuel_mix_sheet(
    file_bytes: bytes,
    sheet: str,
    fuel: str,
) -> list[tuple[pd.Timestamp, float]]:
    """Return (interval_start_cst, mw) pairs for one month sheet and fuel type."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=0)
    df = df[df["Settlement Type"] == "FINAL"]
    rows = df[df["Fuel"] == fuel]

    # Exclude DST repeat columns (e.g. "01:15 (DST)") present in the November sheet
    time_cols = [
        c for c in df.columns
        if c not in ("Date", "Fuel", "Settlement Type", "Total")
        and "(DST)" not in str(c)
    ]
    n = len(time_cols)

    records = []
    for _, row in rows.iterrows():
        date = pd.Timestamp(row["Date"]).normalize()
        for i, col in enumerate(time_cols):
            # Each column label is the interval END time.
            # Interval START = END - 15 min.
            # Last column "0:00" means end-of-day midnight (23:45–00:00).
            if i == n - 1 and col == "0:00":
                start_ts = date + pd.Timedelta(hours=23, minutes=45)
            else:
                h, m = map(int, col.split(":"))
                end_ts = date + pd.Timedelta(hours=h, minutes=m)
                start_ts = end_ts - pd.Timedelta(minutes=15)
            val = row[col]
            records.append((start_ts, float(val) if pd.notna(val) else 0.0))
    return records


def fetch_solar_wind() -> tuple[pd.Series, pd.Series]:
    """Return (solar_cf, wind_cf) hourly series for 2024, UTC-indexed, values in [0, 1]."""
    print("Downloading ERCOT fuel mix Excel (51 MB)...")
    r = requests.get(FUEL_MIX_URL, timeout=180)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("IntGenbyFuel2024.xlsx") as fh:
        file_bytes = fh.read()

    solar_records: list[tuple[pd.Timestamp, float]] = []
    wind_records: list[tuple[pd.Timestamp, float]] = []

    for sheet in MONTH_SHEETS:
        print(f"  Parsing {sheet}...")
        solar_records.extend(_parse_fuel_mix_sheet(file_bytes, sheet, "Solar"))
        wind_records.extend(_parse_fuel_mix_sheet(file_bytes, sheet, "Wind"))

    def to_hourly_cf(records: list[tuple[pd.Timestamp, float]]) -> pd.Series:
        tmp = pd.DataFrame(records, columns=["ts", "mw"])
        tmp = tmp.groupby("ts")["mw"].mean()  # resolve any duplicates
        tmp = tmp.sort_index()
        # Resample 15-min → hourly mean (left-labelled, matching prices convention)
        hourly_mw = tmp.resample("h").mean()
        # Normalize to [0, 1] capacity factor using 2024 peak generation
        peak = hourly_mw.max()
        cf = (hourly_mw / peak).clip(0, 1)
        # ERCOT market data is CST (UTC-6) year-round; convert to UTC
        cf.index = cf.index.tz_localize("Etc/GMT+6").tz_convert("UTC")
        cf.index.name = "datetime_utc"
        return cf

    solar_cf = to_hourly_cf(solar_records)
    solar_cf.name = "solar_cf"
    wind_cf = to_hourly_cf(wind_records)
    wind_cf.name = "wind_cf"
    return solar_cf, wind_cf


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    prices = fetch_rtm_prices()
    prices_path = os.path.join(out_dir, "prices_2024.csv")
    prices.to_csv(prices_path)
    print(f"\nSaved {len(prices)} price rows → {prices_path}")
    print(f"  Range: {prices.min():.2f} – {prices.max():.2f} $/MWh")
    print(f"  Mean:  {prices.mean():.2f} $/MWh")

    solar_cf, wind_cf = fetch_solar_wind()

    solar_path = os.path.join(out_dir, "solar_cf_2024.csv")
    solar_cf.to_csv(solar_path)
    print(f"\nSaved {len(solar_cf)} solar CF rows → {solar_path}")
    print(f"  Peak CF: {solar_cf.max():.4f}  (should be 1.0)")
    print(f"  Mean CF: {solar_cf.mean():.4f}")

    wind_path = os.path.join(out_dir, "wind_cf_2024.csv")
    wind_cf.to_csv(wind_path)
    print(f"\nSaved {len(wind_cf)} wind CF rows → {wind_path}")
    print(f"  Peak CF: {wind_cf.max():.4f}  (should be 1.0)")
    print(f"  Mean CF: {wind_cf.mean():.4f}")

    # Quick alignment check
    common = prices.index.intersection(solar_cf.index).intersection(wind_cf.index)
    print(f"\nTimestamp alignment check: {len(common)} common UTC hours")
    print(f"  Expected: 8784 (2024 is a leap year)")
