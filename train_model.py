from flask import Flask, request, jsonify
from preprocessing import preprocess_data, fit_tokenizer, tokenize_and_pad
from model import build_model, train_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
TEST_SPLIT_SIZE = 0.3
MODEL_SAVE_PATH = './model/hate_speech_model.h5'
DATA_PATH = './datasets/lithuanian/train_tweets_lt.csv'

def main():
    # Load and preprocess the dataset
    train_data = pd.read_csv(DATA_PATH)
    train_data = preprocess_data(train_data, 'tweet')

    # Split the dataset into training and validation sets
    X = train_data['tweet']
    y = train_data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=0)

    # Tokenization and padding
    fit_tokenizer(X_train)
    X_train_pad = tokenize_and_pad(X_train)
    X_val_pad = tokenize_and_pad(X_val)

    # Build and train the model
    model = build_model()
    model = train_model(model, X_train_pad, y_train, X_val_pad, y_val)

    # Save the trained model to a file
    model.save(MODEL_SAVE_PATH)