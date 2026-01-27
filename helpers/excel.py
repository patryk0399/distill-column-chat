# read_distillation_timeseries.py
# Boilerplate to read the generated .xlsx time series with correct dtypes.

from pathlib import Path
import pandas as pd

FILEPATH = Path("./helpers/timeseries/distillation_column_timeseries_2025_5min.xlsx")

def read_timeseries(path: Path = FILEPATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    # Read Excel
    df = pd.read_excel(
        path,
        sheet_name="Timeseries",
        engine="openpyxl",
        parse_dates=["Timestamp"],  # ensure datetime dtype
        dtype={
            "ColumnTag": "string",
            "Mode": "string",
            "AlarmState": "string",
        },
    )

    # enforce numeric types 
    numeric_cols = [
        "FeedFlow_m3ph", "FeedTemp_C", "FeedPressure_barg", "ColumnPressure_bara",
        "TopTemp_C", "BottomTemp_C", "TrayTemp_05_C", "TrayTemp_15_C", "TrayTemp_25_C",
        "RefluxRatio", "RefluxFlow_m3ph", "DistillateFlow_m3ph", "BottomsFlow_m3ph",
        "ReboilerDuty_MW", "CondenserDuty_MW", "RefluxDrumLevel_pct", "SumpLevel_pct",
        "ColumnDP_bar", "xD_LightKey", "xB_LightKey", "ValveReflux_pct", "ValveSteam_pct",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort and set index
    df = df.sort_values("Timestamp").set_index("Timestamp")

    return df


if __name__ == "__main__":
    df = read_timeseries()
    print(df.info())
    print(df.head())