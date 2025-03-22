#!/usr/bin/env python3
"""
train_bert_experiment.py

Example script that:
 - Loads the same CSV dataset specified in train_model.py (reddit_data_annotated.csv).
 - Uses the same 'body' column for text and 'is_hate_speech' for labels.
 - Applies your full custom preprocessing (preprocess_data).
 - Fine-tunes a BERT model (via Hugging Face Transformers) for binary classification.
 - Saves the trained model, tokenizer, classification report, and train history to disk.

Requires:
  - transformers
  - tensorflow 2.x
  - scikit-learn
  - pandas
  - your existing 'preprocessing.py' with preprocess_data
Usage:
  python train_bert_experiment.py
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import logging as hf_logging

# Your custom preprocessing module
from preprocessing import preprocess_data

# Silence some logging from Transformers
hf_logging.set_verbosity_error()

# Constants referencing your data pipeline from train_model.py:
DATA_PATH = './datasets/v2/reddit_data_annotated.csv'
TEXT_COLUMN = 'body'           # same as your code
LABEL_COLUMN = 'is_hate_speech'
TEST_SPLIT_SIZE = 0.2
MODEL_SAVE_PATH = './bert_experiment_output'
BERT_MODEL_NAME = 'bert-base-multilingual-cased'  # or any other

# Training hyperparams
BATCH_SIZE = 8
EPOCHS = 2
MAX_LEN = 128  # typical max subword length for BERT sequences
LEARNING_RATE = 2e-5

def encode_texts(tokenizer, texts, labels, max_len=128):
    """
    Convert text + labels into tf.data.Dataset with BERT encodings.
    """
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset

def main():
    # -----------------------------------------------------------------------
    # 1. Load your CSV dataset
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    # The CSV should have 'body' for text, 'is_hate_speech' for labels,
    # matching your train_model.py

    # -----------------------------------------------------------------------
    # 2. Run your custom preprocessing (same approach as train_model.py)
    # -----------------------------------------------------------------------
    df = preprocess_data(df, TEXT_COLUMN)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # -----------------------------------------------------------------------
    # 3. Train/validation split
    # -----------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=TEST_SPLIT_SIZE,
        random_state=42
    )

    # -----------------------------------------------------------------------
    # 4. Load a BERT tokenizer
    # -----------------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # -----------------------------------------------------------------------
    # 5. Encode data into TF Datasets
    # -----------------------------------------------------------------------
    train_dataset = encode_texts(tokenizer, X_train, y_train, max_len=MAX_LEN)
    val_dataset   = encode_texts(tokenizer, X_val,   y_val,   max_len=MAX_LEN)

    # Shuffle, batch, prefetch
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(1)
    val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(1)

    # -----------------------------------------------------------------------
    # 6. Initialize BERT model for binary classification
    # -----------------------------------------------------------------------
    model = TFBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=2  # 2 classes: hate or not hate
    )

    # -----------------------------------------------------------------------
    # 7. Compile
    # -----------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # -----------------------------------------------------------------------
    # 8. Train
    # -----------------------------------------------------------------------
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset
    )

    # -----------------------------------------------------------------------
    # 9. Make sure output directory exists & save model + tokenizer
    # -----------------------------------------------------------------------
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Also save train history to JSON
    history_file = os.path.join(MODEL_SAVE_PATH, 'train_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history.history, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # 10. Evaluate, print + save classification report
    # -----------------------------------------------------------------------
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}\n")

    # Predict on validation for classification report
    val_texts_list = X_val.tolist()
    encodings = tokenizer(
        val_texts_list,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )
    outputs = model(encodings)
    preds = tf.math.argmax(outputs.logits, axis=1).numpy()

    report = classification_report(y_val, preds, digits=4)
    print("Classification Report (Validation):")
    print(report)

    # Save report to file
    # If you want a machine-readable format, you can do output_dict=True
    # and then store that dict as JSON. Example below shows text file plus JSON.
    report_text_path = os.path.join(MODEL_SAVE_PATH, 'classification_report.txt')
    with open(report_text_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Optionally a JSON version:
    report_dict = classification_report(y_val, preds, digits=4, output_dict=True)
    report_json_path = os.path.join(MODEL_SAVE_PATH, 'classification_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! Model, tokenizer, train history, and reports are saved in {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
