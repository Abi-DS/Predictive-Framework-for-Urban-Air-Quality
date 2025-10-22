# AQI Forecasting Framework

A Streamlit app that forecasts next-day Air Quality Index (AQI) for Indian cities using a generalized XGBoost model trained on multi-city historical data.

## Features
- **City selection** with dynamic forecasts per city (`app.py`).
- **24-hour ahead AQI prediction** using `saved_model/xgb_model.pkl`.
- **Historical trend chart** per city.
- **Feature importance image** support if `feature_importance.png` is present.

## Project Structure
- `app.py` — Streamlit UI, data loading, prediction, and charts.
- `data/city_day.csv` — Input dataset (Date, City, AQI, and pollutant features).
- `saved_model/xgb_model.pkl` — Trained XGBoost model (loaded via `joblib`).
- `feature_importance.png` — Optional visualization shown in the app.
- `.gitignore` — Git ignore rules.
- `venv/` — Local virtual environment (not required if using system Python).

## Prerequisites
- Python 3.9+ recommended
- Internet not required at runtime (all assets local)

## Quickstart
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy joblib plotly xgboost
   ```
3. Ensure required files are present:
   - Dataset: `data/city_day.csv`
   - Model: `saved_model/xgb_model.pkl`
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Use the city dropdown to view tomorrow's predicted AQI and the historical trend.

## Data Expectations
- CSV at `data/city_day.csv` with at least the following columns:
  - `Date` (e.g., `dd-mm-yyyy` or similar parseable date)
  - `City`
  - `AQI`
  - Additional pollutant/meteorological columns used by the model
- The app forward-fills and sorts data internally and filters by selected city.

## Model Expectations
- `saved_model/xgb_model.pkl` should be a compatible XGBoost model loaded by `joblib`.
- The model is expected to expose `get_booster().feature_names` for input alignment.
- The app engineers the following features at inference time:
  - `AQI_lag1`, `AQI_lag2`, `AQI_rolling_mean_7`, `Month`, `DayOfYear`

## Troubleshooting
- If you see a missing file error, verify paths:
  - Dataset: `data/city_day.csv`
  - Model: `saved_model/xgb_model.pkl`
- If the city list is empty, ensure each city has > 365 rows in the dataset.
- If `feature_importance.png` is missing, the app will still run (image is optional).

## Credits
- Team: Harini G (22BDS0085), Abinanthan S (22BDS0122)

## License
This project is for educational purposes. Add a license if you plan to distribute.
