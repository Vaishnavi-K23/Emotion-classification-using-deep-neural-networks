import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy
from sklearn.metrics import f1_score

# --------- CONFIG ---------
MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 128        
BATCH_SIZE = 32         
EMBED_DIM = 256         
LSTM_UNITS_1 = 96       
LSTM_UNITS_2 = 48
WEIGHTS_PATH = "bilstm.weights.h5"
# ---------------------------

# tokenizer only (NO transformer model here)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    """Convert text to token IDs."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def build_model(num_labels: int) -> tf.keras.Model:
    """Embedding + BiLSTM + Dense classifier."""
    input_ids = tf.keras.Input(
        shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids"
    )

    # Large vocab, but that's fine
    x = tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="embedding",
    )(input_ids)

    x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True)
    )(x)
    x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(LSTM_UNITS_2)
    )(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_ids, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)],
    )

    return model


def train(train_path="data\train.csv", dev_path="data/dev.csv"):
    # load the CSVs into Huggingface datasets
    hf_dataset = datasets.load_dataset(
        "csv",
        data_files={"train": train_path, "validation": dev_path},
    )

    # the labels are the names of all columns except the first ('text')
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Collect label columns into a list of floats (0/1)."""
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # TF datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=BATCH_SIZE,
    )

    model = build_model(num_labels=len(labels))

    callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_score",
        mode="max",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_f1_score",
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=1,
    ),
    ]

    # Train – you can bump epochs to 10 if it's still fast enough
    model.fit(
        train_dataset,
        epochs=8,
        validation_data=dev_dataset,
        callbacks=callbacks,
    )

    # Save only weights (simpler & robust)
    model.save_weights(WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


def load_model_with_weights(num_labels: int) -> tf.keras.Model:
    """Rebuild the BiLSTM model architecture and load saved weights."""
    model = build_model(num_labels)
    model.load_weights(WEIGHTS_PATH)
    return model


def tune_threshold(dev_path="data/dev.csv"):
    print("Loading dev data and model weights for threshold tuning...")

    df = pandas.read_csv(dev_path)
    true_labels = df.iloc[:, 1:].values  # 7 label columns

    num_labels = true_labels.shape[1]
    model = load_model_with_weights(num_labels)

    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=BATCH_SIZE,
    )

    probs = model.predict(tf_dataset)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in numpy.linspace(0.2, 0.6, 9):  # 0.20, 0.25, ..., 0.60
        preds = (probs > thr).astype(int)
        f1 = f1_score(true_labels, preds, average="micro")
        print(f"threshold={thr:.2f} -> micro F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print("\nBest threshold:", best_thr)
    print("Best dev micro F1:", best_f1)
    print("👉 Use this threshold in predict().")


def predict(input_path, threshold):
    print("Loading data and model weights for prediction...")

    df = pandas.read_csv(input_path)
    num_labels = df.shape[1] - 1

    model = load_model_with_weights(num_labels)

    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",  # ✅ FIXED
        batch_size=BATCH_SIZE,
    )

    probs = model.predict(tf_dataset)
    predictions = numpy.where(probs > threshold, 1, 0)

    df.iloc[:, 1:] = predictions

    df.to_csv(
        "submission.zip",
        index=False,
        compression=dict(method="zip", archive_name="submission.csv"),
    )

    print("Wrote submission.zip")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict", "tune"})
    parser.add_argument("--input", default="data/dev.csv")
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "tune":
        tune_threshold()
    elif args.command == "predict":
        predict(input_path=args.input, threshold=args.threshold)

