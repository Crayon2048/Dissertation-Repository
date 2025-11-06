# ---------------------------------------------------------------
# Injury Prediction Dataset - Model Comparison Script
# Compatible with Visual Studio Code / Standard Python 3
# ---------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [10, 5]

# ---------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------
# Update this path if needed (for local testing)
data_path = 'C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/injury_data.csv'


if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please check the file path.")

df = pd.read_csv(data_path)

# ---------------------------------------------------------------
# Basic EDA
# ---------------------------------------------------------------
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Description ---")
print(df.describe())

# Missing values heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
plt.title('Missing Data: Training Set')
plt.show()

# ---------------------------------------------------------------
# Split Features and Target
# ---------------------------------------------------------------
if 'Likelihood_of_Injury' not in df.columns:
    raise KeyError("'Likelihood_of_Injury' column not found in dataset.")

x = df.drop('Likelihood_of_Injury', axis=1)
y = df['Likelihood_of_Injury']

# Standardize features
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.1, random_state=101)

# ---------------------------------------------------------------
# Linear Regression Model
# ---------------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# Predictions
y_pred = lin_reg.predict(x_test)

# Plot actual vs predicted
sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Ideal Line')
plt.legend()
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

# Combine actual and predicted values
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\n--- Linear Regression Predictions ---")
print(results.head())

# Residuals
residuals = y_test.values - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.show()

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nLinear Regression Model")
print("--" * 30)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# ---------------------------------------------------------------
# Decision Tree Regressor
# ---------------------------------------------------------------
dt_regressor = DecisionTreeRegressor(random_state=101)
dt_regressor.fit(x_train, y_train)
y_pred_dt = dt_regressor.predict(x_test)
dt_mse = mean_squared_error(y_test, y_pred_dt)
print(f"\nDecision Tree Regression MSE: {dt_mse}")

# ---------------------------------------------------------------
# Random Forest Regressor
# ---------------------------------------------------------------
rf_regressor = RandomForestRegressor(random_state=101)
rf_regressor.fit(x_train, y_train)
y_pred_rf = rf_regressor.predict(x_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Regression MSE: {rf_mse}")

# ---------------------------------------------------------------
# Gradient Boosting Regressor
# ---------------------------------------------------------------
gb_regressor = GradientBoostingRegressor(random_state=101)
gb_regressor.fit(x_train, y_train)
y_pred_gb = gb_regressor.predict(x_test)
gb_mse = mean_squared_error(y_test, y_pred_gb)
print(f"Gradient Boosting Regression MSE: {gb_mse}")

# ---------------------------------------------------------------
# Model Ranking
# ---------------------------------------------------------------
model_scores = {
    "Linear Regression": mse,
    "Decision Tree": dt_mse,
    "Random Forest": rf_mse,
    "Gradient Boosting": gb_mse
}

sorted_scores = sorted(model_scores.items(), key=lambda x: x[1])

print("\n--- Model Rankings (Lower MSE = Better) ---")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score:.6f}")
