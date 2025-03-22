#!/usr/bin/env python3
"""
train_litlat_bert_no_preprocessing.py

Trains the LitLat BERT model ("EMBEDDIA/litlat-bert") as a binary classifier
with *no text preprocessing*. We pass ignore_mismatched_sizes=True to 
TFBertForSequenceClassification, because the base checkpoint does not come 
with a classification head by default.

Outputs:
 - Fine-tuned model weights
 - Tokenizer
 - Train history (train_history.json)
 - Classification report (classification_report.txt + .json)

to ./litlat_bert_experiment_output_no_preprocessing/.

Usage:
  python train_litlat_bert_no_preprocessing.py
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

# Silence some verbose logs
hf_logging.set_verbosity_error()

# ----------------------------- CONFIG -----------------------------------
DATA_PATH = './datasets/v2/reddit_data_annotated.csv'
TEXT_COLUMN = 'body'
LABEL_COLUMN = 'is_hate_speech'
TEST_SPLIT_SIZE = 0.2

# The official Hugging Face checkpoint for LitLat BERT
LITLAT_BERT_CHECKPOINT = 'EMBEDDIA/litlat-bert'

# We'll store all outputs here
MODEL_SAVE_PATH = './litlat_bert_experiment_output_no_preprocessing'

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 2
MAX_LEN = 128
LEARNING_RATE = 2e-5

# ------------------------- HELPER FUNCTIONS -----------------------------
def encode_texts(tokenizer, texts, labels, max_len=128):
    """
    Convert text + labels into a tf.data.Dataset using the BERT tokenizer.
    No text preprocessing: the raw text is tokenized as is.
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

# ------------------------- MAIN TRAIN SCRIPT ----------------------------
def main():
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)
    # We'll assume "body" and "is_hate_speech" columns exist

    # 2) No preprocessing: text is used as-is
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # 3) Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=TEST_SPLIT_SIZE,
        random_state=42
    )

    # 4) Load LitLat BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        "./litlat-bert/litlat-bert/tokenizer_config.json", 
        do_lower_case=False  # or True, your choice
    )
    # 5) Encode data => tf.dataset
    train_dataset = encode_texts(tokenizer, X_train, y_train, max_len=MAX_LEN)
    val_dataset   = encode_texts(tokenizer, X_val,   y_val,   max_len=MAX_LEN)

    # Shuffle, batch, prefetch
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(1)
    val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(1)

    # 6) Load a TFBertForSequenceClassification, ignoring mismatch
    model = TFBertForSequenceClassification.from_pretrained(
        "./litlat-bert/litlat-bert",
        from_pt=True,              # convert PyTorch weights -> TF
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # 7) Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # 8) Train
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset
    )

    # 9) Save all results
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Save model weights + config
    model.save_pretrained(MODEL_SAVE_PATH)
    # Save the tokenizer
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Save training history
    history_file = os.path.join(MODEL_SAVE_PATH, 'train_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history.history, f, indent=2, ensure_ascii=False)

    # 10) Evaluate + classification report
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}\n")

    val_texts = X_val.tolist()
    encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )
    outputs = model(encodings)
    preds = tf.math.argmax(outputs.logits, axis=1).numpy()

    # Print classification report
    report = classification_report(y_val, preds, digits=4)
    print("Classification Report (Validation):")
    print(report)

    # Save classification report (text + JSON)
    report_txt_path = os.path.join(MODEL_SAVE_PATH, 'classification_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report)

    report_dict = classification_report(y_val, preds, digits=4, output_dict=True)
    report_json_path = os.path.join(MODEL_SAVE_PATH, 'classification_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! LitLat BERT (no preprocessing) is in {MODEL_SAVE_PATH}")

# ------------------------- ENTRY POINT -----------------------------------
if __name__ == "__main__":
    main()
