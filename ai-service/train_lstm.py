
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import (
    compute_class_weight,
)

from movement_features import (
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "datasets" / "movements"

WLASL_DIR = (
    BASE_DIR / "datasets" / "wlasl" / "converted"
)

ORIGINAL_TEST_DIR = (
    BASE_DIR / "datasets" / "movements_test"
)

MODEL_DIR = (
    BASE_DIR / "models" / "movement_model_candidate"
)

RESULTS_DIR = (
    BASE_DIR / "results" / "movement_candidate"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 100
BATCH_SIZE = 8

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# LOAD NUMPY FILE
# ==========================================

def load_sequence(path):
    try:
        sequence = np.load(
            path,
            allow_pickle=False,
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):
            print(
                f"Skipping {path}: invalid shape "
                f"{sequence.shape}"
            )
            return None

        if not np.isfinite(sequence).all():
            print(
                f"Skipping {path}: non-finite values"
            )
            return None

        return sequence.astype(np.float32)

    except Exception as error:
        print(f"Skipping {path}: {error}")
        return None


# ==========================================
# LOAD SPLIT
# ==========================================

def load_split(directory, classes):
    X = []
    y = []
    paths = []

    for class_index, sign_name in enumerate(classes):
        sign_dir = directory / sign_name

        if not sign_dir.exists():
            continue

        for path in sorted(sign_dir.glob("*.npy")):
            sequence = load_sequence(path)

            if sequence is None:
                continue

            X.append(sequence)
            y.append(class_index)
            paths.append(path)

    if not X:
        return (
            np.empty(
                (0, SEQUENCE_LENGTH, FEATURE_COUNT),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int32),
            [],
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        paths,
    )


# ==========================================
# DATA AUGMENTATION
# ==========================================

def augment_sequence(sequence, rng):
    augmented = sequence.copy()

    # Small landmark noise
    noise = rng.normal(
        loc=0.0,
        scale=0.005,
        size=augmented.shape,
    ).astype(np.float32)

    # Avoid perturbing completely empty frames.
    # For populated frames, perturb only features
    # associated with a nonzero hand slot.
    for start in (0, 63):
        end = start + 63

        active = np.any(
            augmented[:, start:end] != 0,
            axis=1,
        )

        augmented[active, start:end] += (
            noise[active, start:end]
        )

    return augmented


def augment_training_data(X, y):
    rng = np.random.default_rng(SEED)

    augmented_X = []
    augmented_y = []

    for sequence, label in zip(X, y):
        augmented_X.append(sequence)
        augmented_y.append(label)

        # One augmented copy per training sample
        augmented_X.append(
            augment_sequence(sequence, rng)
        )
        augmented_y.append(label)

    return (
        np.asarray(augmented_X, dtype=np.float32),
        np.asarray(augmented_y, dtype=np.int32),
    )


# ==========================================
# BUILD MODEL
# ==========================================

def build_model(number_of_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURE_COUNT,
            )
        ),

        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(64),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            64,
            activation="relu",
        ),

        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==========================================
# DATASET VALIDATION
# ==========================================

def check_wlasl_overlap(train_paths, val_paths, test_paths):
    def ids(paths):
        return {
            path.stem.removeprefix("wlasl_")
            for path in paths
            if path.stem.startswith("wlasl_")
            or "wlasl" in path.parts
        }

    train_ids = ids(train_paths)
    val_ids = ids(val_paths)
    test_ids = ids(test_paths)

    overlaps = (
        (train_ids & val_ids)
        | (train_ids & test_ids)
        | (val_ids & test_ids)
    )

    if overlaps:
        raise ValueError(
            "WLASL video IDs appear in multiple splits: "
            + str(sorted(overlaps))
        )


def print_counts(name, y, classes):
    counts = Counter(y.tolist())

    print(f"\n{name}")
    print("-" * 40)

    for index, label in enumerate(classes):
        print(
            f"{label:15} {counts.get(index, 0)}"
        )


# ==========================================
# MAIN TRAINING
# ==========================================

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(TRAIN_DIR)

    classes = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
        and any(folder.glob("*.npy"))
    )

    if len(classes) < 2:
        raise ValueError(
            "At least two classes are required"
        )

    print("\nSignAI Expanded LSTM Training")
    print("=" * 50)
    print("Classes:", classes)

    X_train, y_train, train_paths = load_split(
        TRAIN_DIR,
        classes,
    )

    X_wlasl_val, y_wlasl_val, wlasl_val_paths = (
        load_split(
            WLASL_DIR / "val",
            classes,
        )
    )

    X_wlasl_test, y_wlasl_test, wlasl_test_paths = (
        load_split(
            WLASL_DIR / "test",
            classes,
        )
    )

    X_original_test, y_original_test, original_test_paths = (
        load_split(
            ORIGINAL_TEST_DIR,
            classes,
        )
    )

    check_wlasl_overlap(
        train_paths,
        wlasl_val_paths,
        wlasl_test_paths,
    )

    # Use only official WLASL validation examples
    # for validation. Original SignAI test data
    # stays reserved for final evaluation.
    X_val = X_wlasl_val
    y_val = y_wlasl_val

    if len(X_val) == 0:
        raise ValueError(
            "No WLASL validation sequences found"
        )

    # Combine separate test datasets
    test_parts_X = []
    test_parts_y = []

    if len(X_wlasl_test):
        test_parts_X.append(X_wlasl_test)
        test_parts_y.append(y_wlasl_test)

    if len(X_original_test):
        test_parts_X.append(X_original_test)
        test_parts_y.append(y_original_test)

    if not test_parts_X:
        raise ValueError(
            "No independent test sequences found"
        )

    X_test = np.concatenate(
        test_parts_X,
        axis=0,
    )

    y_test = np.concatenate(
        test_parts_y,
        axis=0,
    )

    print_counts(
        "Training distribution",
        y_train,
        classes,
    )

    print_counts(
        "Validation distribution",
        y_val,
        classes,
    )

    print_counts(
        "Testing distribution",
        y_test,
        classes,
    )

    # All classes must exist in training data
    counts = np.bincount(
        y_train,
        minlength=len(classes),
    )

    if np.any(counts == 0):
        raise ValueError(
            "One or more classes have no training data"
        )

    # Warn about tiny classes
    for index, count in enumerate(counts):
        if count < 10:
            print(
                f"WARNING: {classes[index]} has "
                f"only {count} training sequences"
            )

    # ======================================
    # AUGMENT TRAINING DATA
    # ======================================

    X_train_aug, y_train_aug = (
        augment_training_data(
            X_train,
            y_train,
        )
    )

    # ======================================
    # CLASS WEIGHTS
    # ======================================

    class_indices = np.arange(len(classes))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=y_train_aug,
    )

    class_weights = {
        int(index): float(weight)
        for index, weight in enumerate(weights)
    }

    print("\nClass weights:", class_weights)

    # ======================================
    # TRAIN MODEL
    # ======================================

    model = build_model(len(classes))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
        ),
    ]

    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ======================================
    # EVALUATION
    # ======================================

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    probabilities = model.predict(
        X_test,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(classes))),
    )

    print(
        f"\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix")
    print(matrix)

    # ======================================
    # SAVE CANDIDATE
    # ======================================

    model_path = MODEL_DIR / "movement_lstm.keras"
    labels_path = MODEL_DIR / "labels.json"

    model.save(str(model_path))

    with open(
        labels_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(classes, file, indent=2)

    with open(
        MODEL_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )

    with open(
        RESULTS_DIR / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "classes": classes,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
            },
            file,
            indent=2,
        )

    print("\nTraining completed!")
    print("Candidate model:", model_path)
    print("Labels:", labels_path)
    print(
        "Evaluation:",
        RESULTS_DIR / "evaluation.json",
    )

    print(
        "\nYour existing movement_model "
        "has NOT been overwritten."
    )


if __name__ == "__main__":
    main()
