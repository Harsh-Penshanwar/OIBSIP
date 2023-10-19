# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a DataFrame with the dataset
data = pd.DataFrame({
    'Year': [2019, 2020, 2018, 2022, 2017],
    'Mileage': [15000, 10000, 20000, 8000, 30000],
    'Horsepower': [200, 250, 180, 280, 170],
    'MSRP': [35000, 42000, 30000, 45000, 28000]
})

# Add more data for testing
test_data = pd.DataFrame({
    'Year': [2021, 2024],
    'Mileage': [12000, 9000],
    'Horsepower': [210, 260],
    'MSRP': [37000, 43000]
})

data = pd.concat([data, test_data], ignore_index=True)

# Select relevant features
features = data[['Year', 'Mileage', 'Horsepower']]

# Target variable
target = data['MSRP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict the price of a new car
new_car = np.array([[2023, 50000, 250]])  # Provide the Year, Mileage, and Horsepower of the new car
predicted_price = model.predict(new_car)
print("Predicted Price for the New Car:", predicted_price[0])
