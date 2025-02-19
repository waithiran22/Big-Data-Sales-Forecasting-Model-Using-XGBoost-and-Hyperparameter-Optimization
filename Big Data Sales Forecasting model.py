# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from datetime import timedelta

# Step 1: Generate Big Data for Sales Forecasting
np.random.seed(42)
date_range = pd.date_range(start="2015-01-01", end="2023-12-31", freq='D')
data_size = len(date_range) * 100  # 100 stores/products per day

dates = np.random.choice(date_range, data_size)
stores = np.random.choice(['Store A', 'Store B', 'Store C', 'Store D', 'Store E'], data_size)
products = np.random.choice(['Product X', 'Product Y', 'Product Z', 'Product W'], data_size)
prices = np.random.uniform(5, 100, data_size)
promotions = np.random.choice([0, 1], data_size, p=[0.6, 0.4])
seasonality = (np.sin(2 * np.pi * pd.to_datetime(dates).dayofyear / 365) + 1) * 0.5
trends = 1 + 0.001 * pd.to_datetime(dates).year  # Slight upward trend over years
base_sales = np.random.randint(50, 500, data_size)

# Complex Sales Formula
sales = base_sales * (1 + promotions * 0.3) * seasonality * trends
sales = np.round(sales).astype(int)

big_data = pd.DataFrame({
    'Date': dates,
    'Store': stores,
    'Product': products,
    'Price': prices,
    'Promotion': promotions,
    'Sales': sales
})

# Save to CSV for external usage
csv_file_path = 'big_sales_data.csv'
big_data.to_csv(csv_file_path, index=False)

# Step 2: Preprocess Data
big_data['Year'] = pd.to_datetime(big_data['Date']).dt.year
big_data['Month'] = pd.to_datetime(big_data['Date']).dt.month
big_data['Day'] = pd.to_datetime(big_data['Date']).dt.day
big_data['DayOfWeek'] = pd.to_datetime(big_data['Date']).dt.dayofweek
big_data['Quarter'] = pd.to_datetime(big_data['Date']).dt.quarter

# Encode categorical variables
label_encoder = LabelEncoder()
big_data['Store'] = label_encoder.fit_transform(big_data['Store'])
big_data['Product'] = label_encoder.fit_transform(big_data['Product'])

# Feature Scaling
scaler = StandardScaler()
big_data['Price'] = scaler.fit_transform(big_data[['Price']])

# Step 3: Feature Selection
X = big_data[['Store', 'Product', 'Price', 'Promotion', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']]
y = big_data['Sales']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training with XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 6: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_xgb_model = grid_search.best_estimator_

# Step 7: Make Predictions
y_pred = best_xgb_model.predict(X_test)

# Step 8: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Step 9: Feature Importance
feature_importance = pd.Series(best_xgb_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance for Sales Forecasting')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(True)
plt.show()

# Step 10: Visualize Results
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test[:100])), y_test[:100], label='Actual Sales', color='blue')
plt.plot(range(len(y_pred[:100])), y_pred[:100], label='Predicted Sales', color='red')
plt.title('Actual vs Predicted Sales (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Step 11: Display Metrics and Results
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared (R²): {r2}")
