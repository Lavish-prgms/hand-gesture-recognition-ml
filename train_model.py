import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset location
DATA_FILE = "data/hand_gestures.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.pkl")

# Check whether the dataset exists
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}. "
        "Please complete data collection first."
    )

# Load dataset
data = pd.read_csv(DATA_FILE)

if "label" not in data.columns:
    raise ValueError("Dataset must contain a 'label' column.")

if data.empty:
    raise ValueError("Dataset is empty. Collect samples first.")

# Remove incomplete rows
data = data.dropna()

# Separate input features and output labels
X = data.drop(columns=["label"])
y = data["label"]

print("\nDataset loaded successfully!")
print("Total samples:", len(data))
print("\nSamples per gesture:")
print(y.value_counts())

# Check that each class has enough samples
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "At least two gesture classes are needed, "
        "with at least two samples in each class."
    )

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining the ML model...")

# Create and train the classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate the model on the held-out test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
))
print("Class order:", list(model.classes_))

# Save the trained model and feature information
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "classes": list(model.classes_)
    },
    MODEL_FILE
)

print(f"\nTrained model saved to: {MODEL_FILE}")