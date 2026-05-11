import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# =========================
# LOAD DATASET
# =========================

file_path = "./Dataset/cleaned_dataset.csv"

df = pd.read_csv(file_path)

print(df.head())

# =========================
# CREATE DATETIME
# =========================

df['datetime'] = pd.to_datetime(
    df['date'] + ' ' + df['time']
)

df = df.sort_values('datetime')

# =========================
# FEATURES
# =========================

X = df[[
    'pm',
    'MQ135',
    'co',
    'temperature',
    'humidity'
]]

y = df['aqi']

# =========================
# SCALE FEATURES
# =========================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler saved")

# =========================
# TRAIN TEST SPLIT
# =========================

split = int(0.8 * len(X_scaled))

X_train = X_scaled[:split]
X_test = X_scaled[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# BUILD MODEL
# =========================

model = RandomForestRegressor(

    n_estimators=10,

    random_state=42,

    n_jobs=-1
)

# =========================
# TRAIN
# =========================

print("Training Random Forest Model...")

model.fit(
    X_train,
    y_train
)

print("Training Complete")

# =========================
# PREDICT
# =========================

predictions = model.predict(X_test)

# =========================
# EVALUATE
# =========================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

print("MAE:", mae)

print("MSE:", mse)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "aqi_rf_model.pkl"
)

print("Random Forest Model Saved")