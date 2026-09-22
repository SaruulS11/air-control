from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent

csv_path = BASE_DIR / "gesture_dataset.csv"

df = pd.read_csv(csv_path)


print("\nDATASET SHAPE")
print(df.shape)

print("\nGESTURE COUNTS")
print(df["label"].value_counts())


# ---------------------------------------------------------
# FEATURES / LABELS
# ---------------------------------------------------------

X = df.drop(columns=["label"])

y = df["label"]


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

models = {

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        )),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        )),
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=42,
        )),
    ]),
}


# ---------------------------------------------------------
# TRAIN + COMPARE
# ---------------------------------------------------------

results = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.fit(
        X_train,
        y_train,
    )

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
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions,
        )
    )


# ---------------------------------------------------------
# FIND BEST MODEL
# ---------------------------------------------------------

best_name = max(
    results,
    key=lambda name:
        results[name]["accuracy"],
)

best_model = results[best_name]["model"]

best_predictions = (
    results[best_name]["predictions"]
)


print("\n" + "=" * 50)

print(
    f"BEST MODEL: {best_name}"
)

print(
    f"ACCURACY: "
    f"{results[best_name]['accuracy']:.4f}"
)

print("=" * 50)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

model_path = (
    BASE_DIR /
    "gesture_model.pkl"
)

joblib.dump(
    best_model,
    model_path,
)

print(
    f"\nModel saved to:\n{model_path}"
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    xticks_rotation=45,
)

plt.title(
    f"Gesture Recognition - {best_name}"
)

plt.tight_layout()

plt.show()