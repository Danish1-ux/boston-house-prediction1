import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets from local paths
# Replace these paths with the actual file paths on your Windows machine
train_file_path = 'C:/path/to/your/Boston.csv'
test_file_path = 'C:/path/to/your/test.csv'

# Load the datasets into pandas DataFrames
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Display column names to ensure the correct ones are used
print("Train DataFrame Columns:", train_df.columns)
print("Test DataFrame Columns:", test_df.columns)

# Drop 'Unnamed: 0' column if it exists in train_df (it may be an index column from CSV export)
if 'Unnamed: 0' in train_df.columns:
    train_df = train_df.drop(columns=['Unnamed: 0'])

# Separate features (X) and target variable (y) for the training set
X_train = train_df.drop(columns=['medv'])
y_train = train_df['medv']

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Make predictions on the validation set
y_val_pred = model.predict(X_val_split)

# Calculate the Mean Squared Error (MSE) to evaluate model performance
mse = mean_squared_error(y_val_split, y_val_pred)
print(f"Mean Squared Error on Validation Set: {mse}")

# Prepare test data by dropping 'ID' column if it exists
if 'ID' in test_df.columns:
    X_test = test_df.drop(columns=['ID'])
else:
    X_test = test_df

# Generate predictions for the test set
test_predictions = model.predict(X_test)

# Save the predictions to a new CSV file
# Replace 'C:/path/to/your/test_predictions.csv' with the desired output path
output_file_path = 'C:/path/to/your/test_predictions.csv'
test_df['predictions'] = test_predictions
test_df.to_csv(output_file_path, index=False)

print(f"Test predictions saved to {output_file_path}")