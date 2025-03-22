#!/usr/bin/env python3
"""
train_bert_partial_preprocessing.py

Trains a BERT model on the reddit_data_annotated.csv dataset
with PARTIAL (minimal) text preprocessing only:
  - Lowercase
  - Remove special characters
  - (Optional) remove stopwords, etc.
Skipping advanced Lithuanian-specific transformations like ASCII replacement, 
slang expansion, prefix removal, etc.

Outputs (model, tokenizer, classification reports, training history) are 
saved into ./bert_experiment_output_partial_preprocessing/.

Usage:
  python train_bert_partial_preprocessing.py
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

# We'll assume you added a 'preprocess_data_partial' function to your 'preprocessing.py'
# that does minimal steps (lowercasing, remove special chars, maybe stopwords).
from preprocessing import preprocess_data_partial

# Silence some logging from Transformers
hf_logging.set_verbosity_error()

# Constants
DATA_PATH = './datasets/v2/reddit_data_annotated.csv'
TEXT_COLUMN = 'body'           
LABEL_COLUMN = 'is_hate_speech'
TEST_SPLIT_SIZE = 0.2
BERT_MODEL_NAME = 'bert-base-multilingual-cased'  # or any other BERT checkpoint

# Output folder for partial-preprocessing experiment
MODEL_SAVE_PATH = './bert_experiment_output_partial_preprocessing'

# Training hyperparams
BATCH_SIZE = 8
EPOCHS = 2
MAX_LEN = 128   # typical max subword length for BERT sequences
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
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2) PARTIAL preprocessing: no Lithuanian-specific transformations
    #    We assume you defined "preprocess_data_partial" in your preprocessing.py
    #    which does minimal steps (lowercase, remove special chars, maybe stopwords).
    df = preprocess_data_partial(df, TEXT_COLUMN)

    # Convert label column to int if needed
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # 3) Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=TEST_SPLIT_SIZE,
        random_state=42
    )

    # 4) Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 5) Encode data into TF Datasets
    train_dataset = encode_texts(tokenizer, X_train, y_train, max_len=MAX_LEN)
    val_dataset   = encode_texts(tokenizer, X_val,   y_val,   max_len=MAX_LEN)

    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(1)
    val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(1)

    # 6) Initialize TFBertForSequenceClassification
    model = TFBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=2
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

    # 9) Ensure output directory, save model+tokenizer
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Save train history
    history_file = os.path.join(MODEL_SAVE_PATH, 'train_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history.history, f, indent=2, ensure_ascii=False)

    # 10) Evaluate, save classification report
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}\n")

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

    # Save text-based version
    report_text_path = os.path.join(MODEL_SAVE_PATH, 'classification_report.txt')
    with open(report_text_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Save JSON version
    report_dict = classification_report(y_val, preds, digits=4, output_dict=True)
    report_json_path = os.path.join(MODEL_SAVE_PATH, 'classification_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! BERT model (partial preprocessing) is saved in: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
