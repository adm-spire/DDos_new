import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# List of feature CSV files
feature_files = [
    r"dataset\part_1.csv",
    r"dataset\part_2.csv",
    r"dataset\part_3.csv",
    r"C:\Users\rauna\OneDrive\Desktop\DDOS_upgraded\dataset\stripped_custom.csv"
]

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

# Train model sequentially on each dataset
for i, feature_file in enumerate(feature_files, start=1):
    print(f"Training on {feature_file}...")

    # Load dataset
    df = pd.read_csv(feature_file, low_memory=False)

    # Ensure "Label" column exists
    if "Label" not in df.columns:
        raise ValueError(f"Column 'Label' not found in {feature_file}")

    # Drop unwanted columns
    drop_columns = ["Source IP", "Source Port", "Label"]
    available_columns = [col for col in drop_columns if col in df.columns]
    X = df.drop(columns=available_columns, errors="ignore")  # Features
    y = df["Label"]  # Target

    # Convert all columns to numeric (invalid values become NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace infinities with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns with more than 50% missing values
    X.dropna(axis=1, thresh=int(0.5 * len(X)), inplace=True)

    # Fill remaining missing values with column median
    X.fillna(X.median(), inplace=True)

    # Clip extreme values to prevent float overflow
    X = X.clip(lower=-1e6, upper=1e6)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Convert categorical columns to numerical codes
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes  

    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train or update model
    if i == 1:
        dt_model.fit(X_train, y_train)  # First dataset -> train from scratch
    else:
        dt_model.fit(X_train, y_train)  # Retrain on new data

    # Save model
    joblib.dump(dt_model, "sequential_decision_tree_model.joblib")
    print(f"Model updated and saved as sequential_decision_tree_model.joblib")

    # Predict on test set
    y_pred = dt_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy after training on {feature_file}: {accuracy:.4f}\n")

print("Training complete!")

