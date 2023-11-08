# Import necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings 
import Stemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.callbacks import EarlyStopping
warnings.filterwarnings('ignore')

# constants
stopwords_path = './data/stopwords.txt'
lithuanian_to_ascii_map = {
    'ą': 'a', 'č': 'c', 'ę': 'e', 'ė': 'e', 'į': 'i',
    'š': 's', 'ų': 'u', 'ū': 'u', 'ž': 'z'
}
lithuanian_stemmer = Stemmer.Stemmer('lithuanian')
MAX_NB_WORDS = 10000  # defines the max number of words to keep in the vocabulary
MAX_SEQUENCE_LENGTH = 100  # defines the max length of the sequences

with open(stopwords_path, 'r', encoding='utf-8') as file:
    lithuanian_stopwords = file.read().splitlines()

# Load the dataset
train = pd.read_csv('./datasets/old/train_clean_numeric.csv')

# Display basic information about the dataset
print("Dataset:", train.shape)
print(train.isnull().sum())

def lowercase_text(text):
    return text.lower()

def remove_special_characters(text):
    # Define a regex pattern to match unwanted characters
    pattern = re.compile(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?")
    return re.sub(pattern, '', text)

# A simple regex tokenizer function
def tokenize_text(text):
    # Define a pattern to identify tokens: you may adjust this regex to better suit Lithuanian text specifics
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# Stop words are words which do not contain important significance
def remove_stopwords(text, stopwords_list):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(filtered_words)

def replace_lithuanian_characters_to_ascii(text, character_map):
    # Replace each character using the provided mapping
    return ''.join(character_map.get(char, char) for char in text)

# Aggregate all cleaning functions into one
def clean_text(df, text_field, stopwords_list):
    df[text_field] = df[text_field].apply(lowercase_text)
    df[text_field] = df[text_field].apply(remove_special_characters)
    df[text_field] = df[text_field].apply(remove_stopwords, stopwords_list=stopwords_list)
    return df

def stem_text(df, text_field):
    # Function to stem words in the dataframe
    def stem_words(text):
        return " ".join(lithuanian_stemmer.stemWords(text.split()))
    
    df[text_field] = df[text_field].apply(stem_words)
    return df

def tokenize_text_to_string(text):
    # Tokenize text and then join back into a string
    tokens = tokenize_text(text)
    return ' '.join(tokens)

def preprocess_data(df, text_field):
    df = stem_text(df, text_field)
    df[text_field] = df[text_field].apply(tokenize_text_to_string)    
    return df

# Clean the tweets
train = clean_text(train, "tweet", lithuanian_stopwords)
train = preprocess_data(train, 'tweet')

# Define features and labels
X = train['tweet']
y = train['label']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=0)

# Tokenization and Sequence Padding
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH)

# LSTM Model Definition
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
model.fit(X_train_pad, y_train, batch_size=32, epochs=2, validation_data=(X_val_pad, y_val), callbacks=callbacks)

# Evaluate the model
scores = model.evaluate(X_val_pad, y_val, verbose=0)
print("Accuracy on validation set: {:.2f}%".format(scores[1] * 100))