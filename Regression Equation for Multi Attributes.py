import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset
data = {
    'Age': [3, 5, 1, 4, 2],
    'Mileage': [40000, 65000, 15000, 50000, 25000],
    'EngineSize': [2.0, 1.8, 3.0, 2.5, 1.6],
    'Doors': [4, 2, 4, 4, 2],
    'Price': [15000, 12000, 22000, 18000, 13000]
}

df = pd.DataFrame(data)

# Split features and target
X = df[['Age', 'Mileage', 'EngineSize', 'Doors']]
y = df['Price']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Model coefficients
print("\nModel Coefficients:")
feature_coefficients = dict(zip(X.columns, model.coef_))
for feature, coef in feature_coefficients.items():
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Predict new car
new_car = pd.DataFrame({
    'Age': [4],
    'Mileage': [55000],
    'EngineSize': [2.2],
    'Doors': [4]
})
predicted_price = model.predict(new_car)
print(f"Predicted price for new car: ${predicted_price[0]:.2f}")

# Model evaluation
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R-squared score: {r2:.4f}")

# Residual analysis
residuals = y - y_pred

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()
