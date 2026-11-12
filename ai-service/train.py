
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================
# PATHS
# ==========================

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "datasets" / "hand_features"
MODEL_PATH = ROOT / "models" / "hand_model"

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================
# SETTINGS
# ==========================

FEATURE_SIZE = 126
MIN_SAMPLES = 20

EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

tf.keras.utils.set_random_seed(RANDOM_STATE)

# ==========================
# LOAD DATASET
# ==========================

data = []
labels = []

print("\nLoading hand datasets...\n")

for file in sorted(DATASET_PATH.glob("*.npy")):

    arr = np.load(file)

    if (
        arr.ndim != 2
        or arr.shape[1] != FEATURE_SIZE
    ):
        print(
            f"Skipping incompatible: {file.name}"
        )
        continue

    arr = arr[
        np.isfinite(arr).all(axis=1)
    ]

    if len(arr) < MIN_SAMPLES:
        print(
            f"Skipping {file.stem}: "
            f"only {len(arr)} samples"
        )
        continue

    print(
        f"{file.stem}: {len(arr)} samples"
    )

    data.extend(arr)

    labels.extend(
        [file.stem] * len(arr)
    )

if not data:
    raise ValueError(
        "No valid datasets found. "
        "Run prepare_kaggle_dataset.py first."
    )

X = np.asarray(
    data,
    dtype=np.float32
)

# ==========================
# LABEL ENCODING
# ==========================

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

NUM_CLASSES = len(
    encoder.classes_
)

if NUM_CLASSES < 2:
    raise ValueError(
        "At least two classes required"
    )

print("\nDATA SHAPE:", X.shape)
print("NUMBER OF CLASSES:", NUM_CLASSES)
print("CLASSES:", encoder.classes_)

# ==========================
# TRAIN / VALIDATION / TEST
# ==========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# ==========================
# MODEL
# ==========================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(FEATURE_SIZE,)
    ),

    tf.keras.layers.Dense(
        512,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# CALLBACKS
# ==========================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        str(
            MODEL_PATH / "best_model.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
]

# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ==========================
# EVALUATE
# ==========================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==========================")
print("TRAINING RESULTS")
print("==========================")

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test loss: {test_loss:.4f}"
)

# ==========================
# SAVE MODEL
# ==========================

model.save(
    str(
        MODEL_PATH / "sign_model.keras"
    )
)

np.save(
    MODEL_PATH / "labels.npy",
    encoder.classes_
)

print("\nMODEL SAVED SUCCESSFULLY")
print("Classes:", encoder.classes_)
