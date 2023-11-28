from flask import Flask, request, jsonify
from preprocessing import preprocess_data, fit_tokenizer, tokenize_and_pad
from model import build_model, train_model
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
import matplotlib.pyplot as plt

# Constants
TEST_SPLIT_SIZE = 0.2
MODEL_SAVE_PATH = './model/hate_speech_model.keras'
DATA_PATH = './datasets/lithuanian/train_tweets_lt.csv'
TOKENIZER_PATH = './model/tokenizer.json'
HISTORY_PATH = './model/history.json'

def main():
    # Load and preprocess the dataset
    train_data = pd.read_csv(DATA_PATH)
    train_data = preprocess_data(train_data, 'tweet')

    # Split the dataset into training and validation sets
    X = train_data['tweet']
    y = train_data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=42)

    # Tokenization and padding
    tokenizer = fit_tokenizer(X_train)
    X_train_pad = tokenize_and_pad(X_train, tokenizer)
    X_val_pad = tokenize_and_pad(X_val, tokenizer)

    # Build and train the model
    model = build_model()
    model, history = train_model(model, X_train_pad, y_train, X_val_pad, y_val, "./model/")

    # Save the trained model and tokenizer to a file
    model.save(MODEL_SAVE_PATH)
    model.summary()

    if os.path.exists(TOKENIZER_PATH):
        os.remove(TOKENIZER_PATH)
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history.history, f)