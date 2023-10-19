import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your sales data from 'data.csv' (replace with your file path)
data = pd.read_csv('data.csv')

# Perform one-hot encoding for categorical features
data = pd.get_dummies(data, columns=['Target Audience', 'Advertising Platform'])

# Split Data
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Make Predictions
new_data = pd.DataFrame({
    'Advertising Spend': [1000],
    'Target Audience_Segment A': [1],
    'Target Audience_Segment B': [0],
    'Target Audience_Segment C': [0],
    'Advertising Platform_Platform X': [1],
    'Advertising Platform_Platform Y': [0],
    'Advertising Platform_Platform Z': [0]
})
predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales[0]}")
