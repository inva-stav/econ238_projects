#!/usr/bin/env python3
"""
Fetch 10 years of ERCOT summer (June–September) data for Task 4.

Years: 2015–2024
Outputs (saved to task4_data/):
  prices_summer.csv    -- RTM HB_BUSAVG prices, $/MWh, hourly UTC
  solar_cf_summer.csv  -- Solar capacity factor [0, 1], hourly UTC
  wind_cf_summer.csv   -- Wind capacity factor [0, 1], hourly UTC

Notes:
  - Prices via gridstatus ERCOT RTM SPP (15-min aggregated to hourly).
  - Solar/wind via ERCOT FuelMixReport_PreviousYears.zip (IntGenByFuelYYYY.xls/x),
    same source as fetch_data.py, parsed for summer months only.
  - Capacity factors normalized by each year's own summer peak, so CF reflects
    resource availability independent of fleet growth — appropriate for a single
    turbine/panel simulation.
  - Summer = months 6–9 (Jun, Jul, Aug, Sep).
"""

import io
import os
import warnings
import zipfile

import gridstatus
import pandas as pd
import requests

warnings.filterwarnings("ignore")

YEARS = list(range(2015, 2025))
SUMMER_MONTHS = {6, 7, 8, 9}
SETTLEMENT_POINT = "HB_BUSAVG"
FUEL_MIX_URL = (
    "https://www.ercot.com/files/docs/2021/03/10/FuelMixReport_PreviousYears.zip"
)
SUMMER_SHEET_NAMES = ["Jun", "Jul", "Aug", "Sep"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "task4_data")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def fetch_prices() -> pd.Series:
    ercot = gridstatus.Ercot()
    frames = []
    for year in YEARS:
        print(f"  Prices {year}...", flush=True)
        try:
            df = ercot.get_rtm_spp(year=year)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue
        hub = df[df["Location"] == SETTLEMENT_POINT].copy()
        hub = hub.set_index("Interval Start")["SPP"]
        hub.index = pd.DatetimeIndex(hub.index).tz_convert("UTC")
        hourly = hub.resample("h").mean()
        summer = hourly[hourly.index.month.isin(SUMMER_MONTHS)]
        frames.append(summer)
        print(f"    {len(summer)} summer hours")
    out = pd.concat(frames).sort_index()
    out.index.name = "datetime_utc"
    out.name = "lmp_usd_mwh"
    return out


# ---------------------------------------------------------------------------
# Solar / wind capacity factors (Excel zip approach from fetch_data.py)
# ---------------------------------------------------------------------------

def _parse_sheet(
    file_bytes: bytes,
    sheet: str,
    fuel: str,
    year: int,
) -> list[tuple[pd.Timestamp, float]]:
    """Return (interval_start_cst, mw) pairs for one month sheet and fuel."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=0)

    # 2015/2016 format: combined "Date-Fuel" column, datetime.time interval headers
    if "Date-Fuel" in df.columns:
        return _parse_sheet_old(df, fuel)

    # 2017+ format: separate Date/Fuel columns, string time headers
    if "Settlement Type" in df.columns:
        df = df[df["Settlement Type"] == "FINAL"]

    rows = df[df["Fuel"] == fuel]
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
            if i == n - 1 and col == "0:00":
                start_ts = date + pd.Timedelta(hours=23, minutes=45)
            else:
                h, m = map(int, str(col).split(":"))
                end_ts = date + pd.Timedelta(hours=h, minutes=m)
                start_ts = end_ts - pd.Timedelta(minutes=15)
            val = row[col]
            records.append((start_ts, float(val) if pd.notna(val) else 0.0))
    return records


def _parse_sheet_old(
    df: pd.DataFrame,
    fuel: str,
) -> list[tuple[pd.Timestamp, float]]:
    """Parser for 2015/2016 format: 'Date-Fuel' column, datetime.time headers."""
    import datetime
    rows = df[df["Date-Fuel"].str.endswith(f"_{fuel}", na=False)]
    time_cols = [c for c in df.columns if isinstance(c, datetime.time)]
    n = len(time_cols)

    records = []
    for _, row in rows.iterrows():
        date_str = str(row["Date-Fuel"]).split("_")[0]  # e.g. "06/01/15"
        date = pd.Timestamp(date_str).normalize()
        for i, col in enumerate(time_cols):
            # col is a datetime.time object representing interval END
            if i == n - 1 and col == datetime.time(0, 0):
                start_ts = date + pd.Timedelta(hours=23, minutes=45)
            else:
                end_ts = date + pd.Timedelta(hours=col.hour, minutes=col.minute)
                start_ts = end_ts - pd.Timedelta(minutes=15)
            val = row[col]
            records.append((start_ts, float(val) if pd.notna(val) else 0.0))
    return records


def _records_to_hourly_cf(
    records: list[tuple[pd.Timestamp, float]],
    year: int,
) -> pd.Series:
    tmp = pd.DataFrame(records, columns=["ts", "mw"])
    tmp = tmp.groupby("ts")["mw"].mean()
    tmp = tmp.sort_index()
    hourly_mw = tmp.resample("h").mean()
    peak = hourly_mw.max()
    cf = (hourly_mw / peak).clip(0, 1)
    cf.index = cf.index.tz_localize("Etc/GMT+6").tz_convert("UTC")
    cf.index.name = "datetime_utc"
    return cf


def fetch_fuel_mix(zip_bytes: bytes) -> tuple[pd.Series, pd.Series]:
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    # Build a map: year -> filename inside zip (case-insensitive match)
    name_map = {name: name for name in z.namelist()}
    year_file: dict[int, str] = {}
    for year in YEARS:
        for fname in name_map:
            if str(year) in fname:
                year_file[year] = fname
                break

    solar_frames, wind_frames = [], []

    for year in YEARS:
        fname = year_file.get(year)
        if fname is None:
            print(f"  Fuel mix {year}... SKIP (no file in zip)")
            continue
        print(f"  Fuel mix {year} ({fname})...", flush=True)
        file_bytes = z.read(fname)

        solar_records: list[tuple[pd.Timestamp, float]] = []
        wind_records:  list[tuple[pd.Timestamp, float]] = []

        # Sheet names vary by year: "Jun" (2017+), "Jun15" (2015), "Jun16" (2016)
        suffix = str(year)[2:] if year <= 2016 else ""
        for base in SUMMER_SHEET_NAMES:
            sheet = base + suffix
            try:
                solar_records.extend(_parse_sheet(file_bytes, sheet, "Solar", year))
                wind_records.extend(_parse_sheet(file_bytes, sheet, "Wind",  year))
            except Exception as e:
                print(f"    WARN {sheet}: {e}")

        if not solar_records:
            print(f"    SKIP: no solar records parsed")
            continue

        solar_cf = _records_to_hourly_cf(solar_records, year)
        wind_cf  = _records_to_hourly_cf(wind_records,  year)
        solar_frames.append(solar_cf)
        wind_frames.append(wind_cf)
        print(f"    {len(solar_cf)} summer hours")

    solar = pd.concat(solar_frames).sort_index()
    wind  = pd.concat(wind_frames).sort_index()
    solar.name = "solar_cf"
    wind.name  = "wind_cf"
    return solar, wind


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Fetching RTM prices — {YEARS[0]}–{YEARS[-1]}, June–September")
    prices = fetch_prices()
    path = os.path.join(OUT_DIR, "prices_summer.csv")
    prices.to_csv(path)
    print(f"Saved {len(prices):,} rows → {path}")
    print(f"  Range: {prices.min():.2f} – {prices.max():.2f} $/MWh")
    print(f"  Mean:  {prices.mean():.2f} $/MWh\n")

    print(f"Downloading ERCOT fuel mix zip...")
    r = requests.get(FUEL_MIX_URL, timeout=180)
    r.raise_for_status()
    print(f"  {len(r.content) / 1e6:.1f} MB downloaded")

    print(f"\nParsing fuel mix — {YEARS[0]}–{YEARS[-1]}, June–September")
    solar_cf, wind_cf = fetch_fuel_mix(r.content)

    path = os.path.join(OUT_DIR, "solar_cf_summer.csv")
    solar_cf.to_csv(path)
    print(f"Saved {len(solar_cf):,} solar CF rows → {path}")
    print(f"  Mean CF: {solar_cf.mean():.4f}  Peak: {solar_cf.max():.4f}\n")

    path = os.path.join(OUT_DIR, "wind_cf_summer.csv")
    wind_cf.to_csv(path)
    print(f"Saved {len(wind_cf):,} wind CF rows → {path}")
    print(f"  Mean CF: {wind_cf.mean():.4f}  Peak: {wind_cf.max():.4f}\n")

    common = (
        prices.index
        .intersection(solar_cf.index)
        .intersection(wind_cf.index)
    )
    expected = len(YEARS) * (30 + 31 + 31 + 30) * 24
    print(f"Alignment: {len(common):,} common hours  (expected ~{expected:,})")
