import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset
data = {
    "Size (sq.ft.)": [1200, 1500, 1800, 2000, 2500],
    "Price (USD)": [250000, 300000, 350000, 400000, 475000]
}
df = pd.DataFrame(data)

# Features and target variable
X = df[["Size (sq.ft.)"]]
y = df["Price (USD)"]

# Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Regression coefficients
m = model.coef_[0]  # slope
b = model.intercept_  # intercept

# Print the regression equation
print(f"Regression Equation: Price (USD) = {m:.2f} * Size (sq.ft.) + {b:.2f}")

# Get user input for house size
user_size = float(input("Enter the size of the house (in sq.ft.): "))
user_input = pd.DataFrame({"Size (sq.ft.)": [user_size]})

# Predict the price using the model
predicted_price = model.predict(user_input)[0]
print(f"Predicted Price for a house with size {user_size} sq.ft.: ${predicted_price:.2f}")

# Model evaluation metrics
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# Print performance metrics
print(f"R-squared: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")

# Visualization
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Size (sq.ft.)")
plt.ylabel("Price (USD)")
plt.title("House Price Prediction")
plt.legend()
plt.show()
