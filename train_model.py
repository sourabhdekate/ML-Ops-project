import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
print('Columns in dataset:', df.columns.tolist())

# Detect target column
possible_targets = [
    "addiction_level",
    "addicted_level",
    "AddictionRisk",
    "Addiction Risk",
    "addictionrisk",
    "addiction_risk",
    "Risk",
    "risk",
]
target_col = next((c for c in possible_targets if c in df.columns), None)
if target_col is None:
    df.columns = df.columns.str.strip()
    target_col = next((c for c in possible_targets if c in df.columns), None)

if target_col is None:
    raise ValueError(
        "Target column not found. Please validate column names in the CSV and set target_col accordingly."
    )

print('Using target column:', target_col)

# Features and target
# Use only specified features: age, daily_screen_time_hours, sleep_hours
selected_features = ['age', 'daily_screen_time_hours', 'sleep_hours']
X = df[selected_features]
y = df[target_col]

# Drop rows with NaN in target
mask = y.notna()
X = X[mask]
y = y[mask]

# Map target to desired categories
if pd.api.types.is_numeric_dtype(y):
    mapping = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    unique_vals = set(y.dropna().unique())
    if unique_vals <= set(mapping.keys()):
        y = y.map(mapping)
    else:
        print('Numeric target detected, but values are not 0/1/2/3; using raw values.')

print("Target value types:", sorted(y.dropna().unique()))

# No categorical encoding needed for these numeric features
# Fill missing
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")





