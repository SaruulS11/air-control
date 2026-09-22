from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).parent


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    BASE_DIR / "gesture_dataset.csv"
)

validation_df = pd.read_csv(
    BASE_DIR / "gesture_test_dataset.csv"
)

final_test_df = pd.read_csv(
    BASE_DIR / "gesture_final_test.csv"
)


# ============================================================
# CHECK DATA
# ============================================================

print("\nORIGINAL TRAIN:")
print(train_df.shape)

print("\nVALIDATION:")
print(validation_df.shape)

print("\nFINAL TEST:")
print(final_test_df.shape)


print("\nFINAL TEST COUNTS:")
print(
    final_test_df["label"].value_counts()
)


# Make sure all CSV files have exactly the same columns
if not (
    train_df.columns.tolist()
    == validation_df.columns.tolist()
    == final_test_df.columns.tolist()
):
    raise ValueError(
        "CSV column structure does not match!"
    )


# Make sure there are no missing values
print("\nMissing values:")

print(
    "Train:",
    train_df.isna().sum().sum()
)

print(
    "Validation:",
    validation_df.isna().sum().sum()
)

print(
    "Final test:",
    final_test_df.isna().sum().sum()
)


# ============================================================
# COMBINE TRAIN + VALIDATION
# ============================================================

combined_train = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True,
)


print("\nCOMBINED TRAIN:")
print(combined_train.shape)

print(
    combined_train["label"].value_counts()
)


# ============================================================
# FEATURES
# ============================================================

X_train = combined_train.drop(
    columns=["label"]
)

y_train = combined_train["label"]


X_final = final_test_df.drop(
    columns=["label"]
)

y_final = final_test_df["label"]


# ============================================================
# FINAL MLP MODEL
# ============================================================

model = Pipeline([
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
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining final MLP model...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# FINAL TEST
# ============================================================

predictions = model.predict(
    X_final
)


accuracy = accuracy_score(
    y_final,
    predictions
)


print("\n")
print("=" * 60)
print("FINAL UNTOUCHED TEST RESULT")
print("=" * 60)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_final,
        predictions,
        zero_division=0,
    )
)


# ============================================================
# SHOW MISTAKES
# ============================================================

results = final_test_df[
    ["label"]
].copy()

results["prediction"] = predictions


mistakes = results[
    results["label"]
    != results["prediction"]
]


print("\n")
print("=" * 60)
print("MISCLASSIFIED SAMPLES")
print("=" * 60)


if len(mistakes) == 0:
    print(
        "\nNo mistakes! "
        "All final test samples were correct."
    )
else:
    print(mistakes)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y_final.unique()
)

matrix = confusion_matrix(
    y_final,
    predictions,
    labels=labels,
)


matrix_df = pd.DataFrame(
    matrix,
    index=labels,
    columns=labels,
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(matrix_df)