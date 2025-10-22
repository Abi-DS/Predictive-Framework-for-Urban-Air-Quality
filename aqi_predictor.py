# This script trains the model and saves it. Run this file once.

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Starting Model Training ---")

# --- 1. Load and Prepare Data ---
DATA_PATH = 'data/city_day.csv'
MODEL_DIR = 'saved_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.pkl')
CHART_PATH = 'feature_importance.png'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Data Cleaning
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values(['City', 'Date'], inplace=True)
cols_to_fill = [col for col in df.columns if col not in ['City', 'Date']]
df[cols_to_fill] = df.groupby('City')[cols_to_fill].ffill()
df.dropna(inplace=True)
print("Data loaded and cleaned.")

# --- 2. Feature Engineering ---
print("Engineering temporal features...")
df['AQI_lag1'] = df.groupby('City')['AQI'].transform(lambda x: x.shift(1))
df['AQI_lag2'] = df.groupby('City')['AQI'].transform(lambda x: x.shift(2))
df['AQI_rolling_mean_7'] = df.groupby('City')['AQI'].transform(lambda x: x.rolling(window=7).mean())
df['Month'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear
df.dropna(inplace=True)
print("Feature engineering complete.")

# --- 3. Model Training ---
print("Training the XGBoost model...")
y = df['AQI']
X = df.drop(['AQI', 'Date', 'City', 'AQI_Bucket'], axis=1)

final_xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=200, 
    learning_rate=0.08, 
    max_depth=6, 
    random_state=42
)
final_xgb_model.fit(X, y)
print("Model training complete.")

# --- 4. Save the Model and Feature Importance Chart ---
print(f"Saving model to {MODEL_PATH}...")
joblib.dump(final_xgb_model, MODEL_PATH)
print("Model saved successfully.")

feature_importances = pd.DataFrame({
    'feature': X.columns, 
    'importance': final_xgb_model.feature_importances_
})
feature_importances = feature_importances.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('Top 10 Feature Importances (XGBoost)', fontsize=18)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.savefig(CHART_PATH, bbox_inches='tight')
print(f"Feature importance chart saved to {CHART_PATH}.")

