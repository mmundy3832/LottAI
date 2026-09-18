#!/usr/bin/env python3
"""
Fetch historical weather data from Open-Meteo for Austin, TX.
Covers April 2006 through May 2024, daily resolution.
Variables: temperature, humidity, barometric pressure, wind speed.
"""
import requests
import pandas as pd
import time
import os

LATITUDE = 30.2672
LONGITUDE = -97.7431
OUTPUT_FILE = "austin_weather_2006_2024.csv"

# Build year chunks
year_ranges = [("2006-04-01", "2006-12-31")]
for y in range(2007, 2024):
    year_ranges.append((f"{y}-01-01", f"{y}-12-31"))
year_ranges.append(("2024-01-01", "2024-05-31"))

all_frames = []

for start_date, end_date in year_ranges:
    print(f"Fetching {start_date} to {end_date}...")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",
            "pressure_msl_mean",
            "pressure_msl_max",
            "pressure_msl_min",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "wind_speed_10m_max",
            "wind_speed_10m_mean",
            "precipitation_sum",
        ]),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago",
    }

    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    df_chunk = pd.DataFrame({
        "date": daily["time"],
        "temp_max_f": daily["temperature_2m_max"],
        "temp_min_f": daily["temperature_2m_min"],
        "temp_mean_f": daily["temperature_2m_mean"],
        "apparent_temp_max_f": daily["apparent_temperature_max"],
        "apparent_temp_min_f": daily["apparent_temperature_min"],
        "apparent_temp_mean_f": daily["apparent_temperature_mean"],
        "pressure_msl_mean_hpa": daily["pressure_msl_mean"],
        "pressure_msl_max_hpa": daily["pressure_msl_max"],
        "pressure_msl_min_hpa": daily["pressure_msl_min"],
        "humidity_mean_pct": daily["relative_humidity_2m_mean"],
        "humidity_max_pct": daily["relative_humidity_2m_max"],
        "humidity_min_pct": daily["relative_humidity_2m_min"],
        "wind_speed_max_mph": daily["wind_speed_10m_max"],
        "wind_speed_mean_mph": daily["wind_speed_10m_mean"],
        "precipitation_mm": daily["precipitation_sum"],
    })
    all_frames.append(df_chunk)
    time.sleep(0.5)  # be courteous

df = pd.concat(all_frames, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

print(f"\nTotal days: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSample data:")
print(df.head())
print(f"\nColumn stats:")
print(df.describe())

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
