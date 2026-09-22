from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

train_path = BASE_DIR / "gesture_dataset.csv"
test_path = BASE_DIR / "gesture_test_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\nTRAIN DATA:")
print(train_df.shape)

print("\nTEST DATA:")
print(test_df.shape)


# ============================================================
# VERIFY DATA
# ============================================================

expected_columns = train_df.columns.tolist()

if test_df.columns.tolist() != expected_columns:
    raise ValueError(
        "Train and test CSV columns do not match."
    )

print("\nTRAIN CLASSES:")
print(train_df["label"].value_counts())

print("\nTEST CLASSES:")
print(test_df["label"].value_counts())


# ============================================================
# FEATURES + LABELS
# ============================================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


# ============================================================
# MODELS
# ============================================================

models = {

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        ),
    ]),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
        ),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42,
            )
        ),
    ]),

    "MLP": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1500,
                random_state=42,
            )
        ),
    ]),
}


# ============================================================
# TRAIN + TEST ON COMPLETELY NEW DATA
# ============================================================

results = {}

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train ONLY using original 600 samples
    model.fit(
        X_train,
        y_train,
    )

    # Predict the new 200 samples
    predictions = model.predict(
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "predictions": predictions,
    }

    print(
        f"\nExternal accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


# ============================================================
# RESULTS SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

sorted_results = sorted(
    results.items(),
    key=lambda item:
        item[1]["accuracy"],
    reverse=True,
)

for name, result in sorted_results:
    print(
        f"{name:15} : "
        f"{result['accuracy'] * 100:.2f}%"
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_name = sorted_results[0][0]

best_model = results[
    best_name
]["model"]

best_predictions = results[
    best_name
]["predictions"]

best_accuracy = results[
    best_name
]["accuracy"]


print("\n")
print("=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print(
    f"{best_name}"
)

print(
    f"External accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    BASE_DIR /
    "gesture_model.pkl"
)

joblib.dump(
    best_model,
    model_path,
)

print(
    f"\nModel saved to:\n"
    f"{model_path}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    xticks_rotation=45,
    cmap="Blues",
)

plt.title(
    f"{best_name} - New Session Test"
)

plt.tight_layout()

plt.show()