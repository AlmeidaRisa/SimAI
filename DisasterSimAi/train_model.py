import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load your historical flood data
data = pd.read_csv('historical_flood_data.csv')

# Data Preprocessing
# Handle missing values by replacing 'NA' and other non-numeric strings
data['Severity'] = pd.to_numeric(data['Severity'], errors='coerce').fillna(0)
data['Duration(Days)'] = pd.to_numeric(data['Duration(Days)'], errors='coerce').fillna(0)
data['Human fatality'] = pd.to_numeric(data['Human fatality'], errors='coerce').fillna(0)
data['Human Displaced'] = pd.to_numeric(data['Human Displaced'], errors='coerce').fillna(0)

# Create a binary target variable (1 for significant impact, 0 for no significant impact)
data['significant_impact'] = (data['Human Displaced'] > 0).astype(int)

# Prepare features and labels
X = data[['Duration(Days)', 'Human fatality', 'Severity']]
y = data['significant_impact']

# Check for any remaining non-numeric data in features
if X.isnull().values.any():
    print("Warning: There are missing values in the feature set.")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model for use in your Flask app
joblib.dump(model, 'flood_model.pkl')
