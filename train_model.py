import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load all data files
all_files = glob.glob("data_*.csv")
data = []

for file in all_files:
    df = pd.read_csv(file, header=None)
    data.append(df)

# Combine all data
full_data = pd.concat(data)
X = full_data.iloc[:, :-1]  # features: x1, y1, ..., x21, y21
y = full_data.iloc[:, -1]   # label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "asl_model.pkl")  # or "isl_model.pkl"
print("Model saved as asl_model.pkl")
