import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data_none, fit_tokenizer, tokenize_and_pad
from model import build_model, train_model

TEST_SPLIT_SIZE = 0.2
DATA_PATH = './datasets/v2/reddit_data_annotated.csv'

MODEL_SAVE_PATH = './model_no_preprocessing/hate_speech_model.keras'
TOKENIZER_PATH = './model_no_preprocessing/tokenizer.json'
HISTORY_PATH = './model_no_preprocessing/history.json'


def main():
    train_data = pd.read_csv(DATA_PATH)
    train_data = preprocess_data_none(train_data, 'body')

    X = train_data['body']
    y = train_data['is_hate_speech']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=42
    )

    tokenizer = fit_tokenizer(X_train)
    X_train_pad = tokenize_and_pad(X_train, tokenizer)
    X_val_pad   = tokenize_and_pad(X_val, tokenizer)

    model = build_model()
    model, history = train_model(model, X_train_pad, y_train, X_val_pad, y_val, "./model_no_preprocessing/")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
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
        json.dump(history.history, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
