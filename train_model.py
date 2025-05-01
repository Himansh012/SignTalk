'''
import pandas as pd
import string
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 📁 Folder containing your CSV files
DATA_DIR = "data"
all_data = []

print("📂 Looking for CSV files in:", os.path.abspath(DATA_DIR))

for letter in string.ascii_uppercase:
    file_path = os.path.join(DATA_DIR, f"data_{letter}.csv")
    if os.path.exists(file_path):
        print(f"✅ Found: {file_path}")
        df = pd.read_csv(file_path, header=None)
        all_data.append(df)
    else:
        print(f"⚠️ Missing: {file_path} (skipping)")

# ✅ Confirm data was loaded
if not all_data:
    print("❌ No valid CSV files found. Check your data folder and filenames.")
    exit()

# 🧠 Combine all letter data
data = pd.concat(all_data, ignore_index=True)
print(f"\n📊 Loaded {len(data)} samples.")

# Split into features and labels
X = data.iloc[:, :-1]  # landmark coordinates (x1, y1, ..., x21, y21)
y = data.iloc[:, -1]   # labels (A–Z)

# Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Model trained!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "asl_model.pkl")
print("\n💾 Model saved as 'asl_model.pkl'")
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("full_dataset.csv")

# Select only numeric features for training
X = df.select_dtypes(include=['number'])

# Target labels
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ Save the model (STEP 4)
joblib.dump(model, "sign_model.pkl")
print("✅ Model saved successfully as sign_model.pkl")
