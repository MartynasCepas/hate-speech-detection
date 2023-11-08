# Import necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings 
import Stemmer
warnings.filterwarnings('ignore')

# constants
stopwords_path = './data/stopwords.txt'
lithuanian_to_ascii_map = {
    'ą': 'a', 'č': 'c', 'ę': 'e', 'ė': 'e', 'į': 'i',
    'š': 's', 'ų': 'u', 'ū': 'u', 'ž': 'z'
}
lithuanian_stemmer = Stemmer.Stemmer('lithuanian')

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

# Feature extraction: Transform text data into TF-IDF features
vectorizer = TfidfVectorizer(lowercase=True, max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Initialize and train the logistic regression model
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train_tfidf, y_train)

# Predict on the validation set
y_pred_val = logreg.predict(X_val_tfidf)

# Evaluate the model
print("Accuracy on validation set:", accuracy_score(y_val, y_pred_val))
print("Classification Report on validation set:\n", classification_report(y_val, y_pred_val))

# Output the first few rows of the train dataset to confirm preprocessing
print(train.head())