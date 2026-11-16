"""Train a CNN that answers two yes/no questions per image: tlgw? nwb?

Two models you can pick with --model:
  small      a tiny CNN built from scratch (learns everything itself)
  mobilenet  MobileNetV2 pretrained on ImageNet (1.4M photos), then
             fine-tuned on our images. This is TRANSFER LEARNING.

Multi-label setup:
  - the last layer has 2 neurons, one per label
  - each uses a SIGMOID (squashes to 0-1, a probability for that label)
  - loss is binary cross-entropy, applied to each label separately
  (For one-answer-only problems you'd use softmax instead; here an image
   can be both TLGW and NWB, so each label gets its own probability.)

Usage:
  python src/train_cnn.py --model mobilenet --data data/processed --out results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

LABELS = ["tlgw", "nwb"]
IMG_SIZE = 224
SEED = 42


def make_dataset(df, data_dir, training, batch_size=32):
    paths = [str(Path(data_dir) / "images" / f) for f in df.file]
    y = df[LABELS].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(len(df), seed=SEED)

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        if training:
            # Data augmentation: only changes that DON'T change the answer.
            # A mirrored photo still has the same text/background problem.
            # We do NOT change brightness or colour: that could turn a white
            # background grey and silently make the 'nwb' label wrong.
            img = tf.image.random_flip_left_right(img, seed=SEED)
        return img, label  # pixel values 0-255; each model rescales itself

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def small_cnn():
    """A from-scratch CNN: 3 blocks of (convolution -> pooling).

    Convolutions slide small filters over the image to find patterns
    (edges, then shapes, then objects). Pooling shrinks the image so
    later layers see a bigger area.
    """
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)
    for filters in [16, 32, 64]:
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(LABELS), activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="small_cnn"), None


def mobilenet():
    """MobileNetV2 body (frozen at first) + a new 2-neuron head."""
    base = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False  # phase 1: keep the pretrained knowledge fixed
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)  # 0-255 -> -1..1, as MobileNetV2 expects
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(LABELS), activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="mobilenet_v2"), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["small", "mobilenet"], default="mobilenet")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=8, help="max epochs per phase (early stopping may end sooner)")
    ap.add_argument("--fine-tune-layers", type=int, default=30, help="mobilenet: how many top layers to unfreeze in phase 2")
    args = ap.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(Path(args.data) / "labels.csv")
    train_df, val_df, test_df = (labels[labels.split == s] for s in ["train", "val", "test"])
    print(f"train {len(train_df)}  val {len(val_df)}  test {len(test_df)}")
    train_ds = make_dataset(train_df, args.data, training=True)
    val_ds = make_dataset(val_df, args.data, training=False)

    # Stop when the validation loss stops improving, and keep the best weights.
    # This is our defence against OVERFITTING (memorising the training set).
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model, base = small_cnn() if args.model == "small" else mobilenet()
    histories = []

    # Phase 1: train (for mobilenet: only the new head).
    compile_model(model, lr=1e-3)
    h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early], verbose=2)
    histories.append(pd.DataFrame(h.history).assign(phase=1))

    # Phase 2 (mobilenet only): FINE-TUNING. Unfreeze the top layers and train
    # with a 100x smaller learning rate, so we gently adjust the pretrained
    # features to our images instead of wrecking them.
    if base is not None and args.fine_tune_layers > 0:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm frozen: standard practice for small datasets
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, lr=1e-5)
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early], verbose=2)
        histories.append(pd.DataFrame(h.history).assign(phase=2))

    pd.concat(histories, ignore_index=True).to_csv(out / f"history_{args.model}.csv", index_label="epoch")
    model.save(out / f"cnn_{args.model}.keras")

    # Save predicted probabilities for val and test: Step 3 evaluates these.
    rows = []
    for df in (val_df, test_df):
        probs = model.predict(make_dataset(df, args.data, training=False), verbose=0)
        part = df.copy()
        for i, label in enumerate(LABELS):
            part[f"{label}_prob"] = probs[:, i]
            part[f"{label}_pred"] = (probs[:, i] >= 0.5).astype(int)
        rows.append(part)
    preds = pd.concat(rows)
    preds.to_csv(out / f"cnn_{args.model}_predictions.csv", index=False)

    test = preds[preds.split == "test"]
    for label in LABELS:
        print(f"{label}: test accuracy {(test[label] == test[f'{label}_pred']).mean():.3f}")
    print("saved model and predictions to", out)


if __name__ == "__main__":
    main()
