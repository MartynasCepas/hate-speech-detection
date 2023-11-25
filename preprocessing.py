import re
import json
import pandas as pd
import Stemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Define constants
STOPWORDS_PATH = './data/stopwords.txt'
SLANG_PATH = './data/slang.json'
CHAR_MAP_PATH = './data/lithuanian_to_ascii_map.json'

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

lithuanian_stemmer = Stemmer.Stemmer('lithuanian')
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

# Load data from files
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    lithuanian_stopwords = file.read().splitlines()
with open(SLANG_PATH, 'r', encoding='utf-8') as file:
    slang_dict = json.load(file)
with open(CHAR_MAP_PATH, 'r', encoding='utf-8') as file:
    lithuanian_to_ascii_map = json.load(file)
    

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

# Function to replace slang with expansion in text
def replace_slang(text, slang_dictionary):
    # Split the text into words and replace slang where found
    words = text.split()
    replaced_text = ' '.join([slang_dictionary.get(word.lower(), word) for word in words])
    return replaced_text

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

# Aggregate all cleaning functions into one
def clean_text(df, text_field, stopwords_list, slang_dictionary=slang_dict, character_map=lithuanian_to_ascii_map):
    df[text_field] = df[text_field].apply(lowercase_text)
    df[text_field] = df[text_field].apply(replace_lithuanian_characters_to_ascii, character_map=character_map)
    df[text_field] = df[text_field].apply(replace_slang, slang_dictionary=slang_dictionary)
    df[text_field] = df[text_field].apply(remove_special_characters)
    df[text_field] = df[text_field].apply(remove_stopwords, stopwords_list=stopwords_list)
    return df

def preprocess_data(df, text_field):
    df = clean_text(df, text_field, lithuanian_stopwords, slang_dict)
    df = stem_text(df, text_field)
    df[text_field] = df[text_field].apply(tokenize_text_to_string)    
    return df

def fit_tokenizer(texts):
    tokenizer.fit_on_texts(texts)

def tokenize_and_pad(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences
