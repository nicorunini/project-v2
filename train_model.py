import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_DIR = Path("datasets")
DISPLAY_LABELS = ["Closed", "Open", "Partially Closed"]
IMAGE_SIZE = 128

MODEL_NAMES = ["custom_cnn", "efficientnet_b0", "mobilenet_v2", "densenet_121"]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_custom_cnn(image_size: int, class_count: int):
    """Improved custom CNN with separable convs, normalization, and dropout."""
    model = models.Sequential(name="custom_cnn")
    model.add(layers.Input(shape=(image_size, image_size, 3)))
    model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.SeparableConv2D(48, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))

    model.add(layers.SeparableConv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.SeparableConv2D(96, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))

    model.add(layers.SeparableConv2D(128, 3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(96, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(class_count, activation="softmax"))
    return model


def build_efficientnet_b0(image_size: int, class_count: int):
    """EfficientNet-B0 with ImageNet weights, top replaced for fine-tuning."""
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False  # freeze base; unfreeze later for fine-tuning

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnet_b0")


def build_mobilenet_v2(image_size: int, class_count: int):
    """MobileNetV2 with ImageNet weights, top replaced for fine-tuning."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenet_v2")


def build_densenet_121(image_size: int, class_count: int):
    """DenseNet-121 with ImageNet weights, top replaced for fine-tuning."""
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="densenet_121")


MODEL_BUILDERS = {
    "custom_cnn":      build_custom_cnn,
    "efficientnet_b0": build_efficientnet_b0,
    "mobilenet_v2":    build_mobilenet_v2,
    "densenet_121":    build_densenet_121,
}


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def create_generators(args):
    # Transfer learning models expect inputs in [0, 255] range handled
    # internally, but we rescale to [0, 1] which also works fine for all four.
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.validation_split,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.7, 1.3],
        channel_shift_range=15.0,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.validation_split,
    )

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=args.seed,
    )
    validation_data = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )
    return train_data, validation_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def display_labels_from_generator(generator):
    ordered = sorted(generator.class_indices.items(), key=lambda item: item[1])
    return [label.replace("_", " ").title() for label, _ in ordered]


def save_labels(path: Path, labels):
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def print_confusion_matrix(model, validation_data, labels):
    validation_data.reset()
    probabilities = model.predict(validation_data, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = validation_data.classes
    matrix = tf.math.confusion_matrix(
        actual, predicted, num_classes=len(labels)
    ).numpy()

    print("\nConfusion matrix")
    print("Rows = actual, columns = predicted")
    print(" " * 20 + " ".join(f"{label[:8]:>8}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"{label[:18]:<18}  " + " ".join(f"{value:8d}" for value in row))

    print("\nPer-class metrics")
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        actual_total = np.sum(matrix[index, :])
        predicted_total = np.sum(matrix[:, index])
        recall = true_positive / actual_total if actual_total else 0.0
        precision = true_positive / predicted_total if predicted_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        print(f"  {label}: precision {precision:.2%}, recall {recall:.2%}, F1 {f1:.2%}")


def train_and_evaluate(model_name, args, train_data, validation_data, labels):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")

    builder = MODEL_BUILDERS[model_name]
    model = builder(args.image_size, train_data.num_classes)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        ),
    ]

    start = time.time()
    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start

    validation_data.reset()
    loss, accuracy = model.evaluate(validation_data, verbose=0)
    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))

    model_path = Path(args.output_dir) / f"{model_name}.h5"
    model.save(str(model_path))

    print(f"  Saved to        : {model_path}")
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Final val loss  : {loss:.4f}")
    print(f"  Final val acc   : {accuracy:.2%}")
    print(f"  Best val acc    : {best_val_accuracy:.2%}")
    print_confusion_matrix(model, validation_data, labels)

    return {
        "model_name":       model_name,
        "model_path":       str(model_path),
        "val_loss":         round(loss, 4),
        "val_accuracy":     round(accuracy, 4),
        "best_val_accuracy": round(best_val_accuracy, 4),
        "training_time_s":  round(elapsed, 1),
        "history":          history.history,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark eye-state classifiers: Custom CNN vs transfer learning models."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=MODEL_NAMES,
                        help="Which models to train (default: all four)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="trained_models",
                        help="Directory to save .h5 model files")
    parser.add_argument("--labels-path", default="labels.txt")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_data, validation_data = create_generators(args)
    labels = display_labels_from_generator(train_data)
    save_labels(Path(args.labels_path), labels)

    print(f"Classes found   : {', '.join(labels)}")
    print(f"Training samples: {train_data.samples}")
    print(f"Val samples     : {validation_data.samples}")
    print(f"Image size      : {args.image_size}x{args.image_size}")
    print(f"Models to train : {', '.join(args.models)}")

    results = []
    for model_name in args.models:
        result = train_and_evaluate(
            model_name, args, train_data, validation_data, labels
        )
        results.append(result)
        # Reset generators for next model
        train_data.reset()
        validation_data.reset()

    # Summary table
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Val Acc':>8} {'Best Acc':>9} {'Loss':>7} {'Time(s)':>8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['model_name']:<20} "
            f"{r['val_accuracy']:>7.2%} "
            f"{r['best_val_accuracy']:>8.2%} "
            f"{r['val_loss']:>7.4f} "
            f"{r['training_time_s']:>8.1f}"
        )

    best = max(results, key=lambda r: r["best_val_accuracy"])
    print(f"\nBest model: {best['model_name']} ({best['best_val_accuracy']:.2%} best val accuracy)")


if __name__ == "__main__":
    main()
