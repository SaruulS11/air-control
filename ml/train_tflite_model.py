from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

files = [
    BASE_DIR / "gesture_dataset.csv",          # 600
    BASE_DIR / "gesture_test_dataset.csv",     # 200
    BASE_DIR / "gesture_final_test.csv",       # 100
]


# ============================================================
# LOAD ALL 900 SAMPLES
# ============================================================

dataframes = [
    pd.read_csv(path)
    for path in files
]

df = pd.concat(
    dataframes,
    ignore_index=True,
)


print("\nFULL DEPLOYMENT DATASET:")
print(df.shape)

print("\nGESTURE COUNTS:")
print(df["label"].value_counts())


# ============================================================
# CHECK DATA
# ============================================================

if df.isna().sum().sum() != 0:
    raise ValueError(
        "Dataset contains missing values!"
    )


# ============================================================
# LABEL ORDER
# ============================================================

# IMPORTANT:
# This order will also be used later in Flutter.

labels = [
    "open_hand",
    "fist",
    "point",
    "peace",
    "thumbs_up",
    "okay",
    "rock",
    "three",
    "call",
    "gun",
]


label_to_index = {
    label: index
    for index, label in enumerate(labels)
}


unknown_labels = set(
    df["label"].unique()
) - set(labels)

if unknown_labels:
    raise ValueError(
        f"Unexpected labels found: {unknown_labels}"
    )


# ============================================================
# FEATURES
# ============================================================

X = df.drop(
    columns=["label"]
).to_numpy(
    dtype=np.float32
)


y = df["label"].map(
    label_to_index
).to_numpy(
    dtype=np.int32
)


print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nNumber of features:")
print(X.shape[1])


# ============================================================
# NORMALIZATION LAYER
# ============================================================

normalizer = tf.keras.layers.Normalization(
    axis=-1,
    name="feature_normalization",
)

normalizer.adapt(X)


# ============================================================
# BUILD MLP
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(63,),
        name="landmark_features",
    ),

    normalizer,

    tf.keras.layers.Dense(
        128,
        activation="relu",
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu",
    ),

    tf.keras.layers.Dense(
        len(labels),
        activation="softmax",
        name="gesture_probabilities",
    ),
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"],
)


model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nTraining final deployment model...\n")


history = model.fit(
    X,
    y,

    epochs=100,

    batch_size=32,

    shuffle=True,

    verbose=1,
)


# ============================================================
# TRAINING RESULT
# ============================================================

final_loss = history.history["loss"][-1]
final_accuracy = history.history["accuracy"][-1]


print("\n")
print("=" * 60)
print("FINAL DEPLOYMENT MODEL")
print("=" * 60)

print(
    f"Training loss: "
    f"{final_loss:.6f}"
)

print(
    f"Training accuracy: "
    f"{final_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE KERAS MODEL
# ============================================================

keras_path = (
    BASE_DIR /
    "gesture_model.keras"
)

model.save(
    keras_path
)

print(
    f"\nKeras model saved:\n"
    f"{keras_path}"
)


# ============================================================
# CONVERT TO TFLITE
# ============================================================

converter = (
    tf.lite.TFLiteConverter
    .from_keras_model(model)
)

tflite_model = converter.convert()


tflite_path = (
    BASE_DIR /
    "gesture_model.tflite"
)

tflite_path.write_bytes(
    tflite_model
)


print(
    f"\nTFLite model saved:\n"
    f"{tflite_path}"
)


# ============================================================
# SAVE LABELS
# ============================================================

labels_path = (
    BASE_DIR /
    "gesture_labels.txt"
)

labels_path.write_text(
    "\n".join(labels),
    encoding="utf-8",
)


print(
    f"\nLabels saved:\n"
    f"{labels_path}"
)


# ============================================================
# BASIC TEST
# ============================================================

sample = X[0:1]

keras_prediction = model.predict(
    sample,
    verbose=0,
)[0]

predicted_index = int(
    np.argmax(keras_prediction)
)

predicted_label = labels[
    predicted_index
]

confidence = float(
    keras_prediction[
        predicted_index
    ]
)


print("\n")
print("=" * 60)
print("QUICK MODEL CHECK")
print("=" * 60)

print(
    "Actual:",
    df.iloc[0]["label"],
)

print(
    "Predicted:",
    predicted_label,
)

print(
    f"Confidence: "
    f"{confidence * 100:.2f}%"
)