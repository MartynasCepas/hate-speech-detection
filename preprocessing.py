import re
import json
import pandas as pd
import Stemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

# Define constants
STOPWORDS_PATH = './data/stopwords.txt'
SLANG_PATH = './data/slang.json'
CHAR_MAP_PATH = './data/lithuanian_to_ascii_map.json'
TOKENIZER_PATH = './model/tokenizer.json'
LITHUANIAN_PREFIXES_PATH = './data/prefixes.json'
EXPORT_DIR = './exported_steps'

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

lithuanian_stemmer = Stemmer.Stemmer('lithuanian')
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

# Load data from files
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    lithuanian_stopwords = file.read().splitlines()

with open(LITHUANIAN_PREFIXES_PATH, 'r', encoding='utf-8') as file:
    lithuanian_prefixes = file.read().splitlines()

with open(SLANG_PATH, 'r', encoding='utf-8') as file:
    slang_dict = json.load(file)

with open(CHAR_MAP_PATH, 'r', encoding='utf-8') as file:
    lithuanian_to_ascii_map = json.load(file)


def lowercase_text(text):
    return text.lower()

def remove_special_characters(text):
    pattern = re.compile(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?")
    return re.sub(pattern, '', text)

def tokenize_text(text):
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def remove_stopwords(text, stopwords_list):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(filtered_words)

def replace_lithuanian_characters_to_ascii(text, character_map):
    return ''.join(character_map.get(char, char) for char in text)

def replace_slang(text, slang_dictionary):
    words = text.split()
    replaced_text = ' '.join([slang_dictionary.get(word.lower(), word) for word in words])
    return replaced_text

def remove_prefixes(word, prefix_list):
    for prefix in prefix_list:
        if word.startswith(prefix) and len(word) > len(prefix):
            return word[len(prefix):]
    return word

def tokenize_text_to_string(text):
    tokens = tokenize_text(text)
    return ' '.join(tokens)

def stem_text(df, text_field):
    def stem_words(text):
        return " ".join(lithuanian_stemmer.stemWords(text.split()))
    
    df[text_field] = df[text_field].apply(stem_words)
    return df

def clean_text_stepwise(df, text_field):
    # Lowercase
    df[text_field + "_step1_lower"] = df[text_field].apply(lowercase_text)
    df.to_csv(f"{EXPORT_DIR}/step1_lower.csv", index=False, encoding='utf-8')

    # ASCII mapping
    df[text_field + "_step2_ascii"] = df[text_field + "_step1_lower"].apply(
        replace_lithuanian_characters_to_ascii, 
        character_map=lithuanian_to_ascii_map
    )
    df.to_csv(f"{EXPORT_DIR}/step2_ascii.csv", index=False, encoding='utf-8')

    # Slang replacement
    df[text_field + "_step3_slang"] = df[text_field + "_step2_ascii"].apply(
        replace_slang, 
        slang_dictionary=slang_dict
    )
    df.to_csv(f"{EXPORT_DIR}/step3_slang.csv", index=False, encoding='utf-8')

    # Remove special characters
    df[text_field + "_step4_nospecial"] = df[text_field + "_step3_slang"].apply(
        remove_special_characters
    )
    df.to_csv(f"{EXPORT_DIR}/step4_nospecial.csv", index=False, encoding='utf-8')

    # Remove stopwords
    df[text_field + "_step5_nostop"] = df[text_field + "_step4_nospecial"].apply(
        remove_stopwords, 
        stopwords_list=lithuanian_stopwords
    )
    df.to_csv(f"{EXPORT_DIR}/step5_nostop.csv", index=False, encoding='utf-8')

    # Remove prefixes
    df[text_field + "_step6_noprefix"] = df[text_field + "_step5_nostop"].apply(
        lambda x: ' '.join([remove_prefixes(word, lithuanian_prefixes) for word in x.split()])
    )
    df.to_csv(f"{EXPORT_DIR}/step6_noprefix.csv", index=False, encoding='utf-8')

    # Overwrite original column with final transformation from step 6
    df[text_field] = df[text_field + "_step6_noprefix"]
    return df

def preprocess_data(df, text_field):
    # Stepwise cleaning
    df = clean_text_stepwise(df, text_field)
    df.to_csv(f"{EXPORT_DIR}/step6_cleaned.csv", index=False, encoding='utf-8')

    # Stemming
    df = stem_text(df, text_field)
    df.to_csv(f"{EXPORT_DIR}/step7_stemmed.csv", index=False, encoding='utf-8')

    # Tokenize to string
    df[text_field] = df[text_field].apply(tokenize_text_to_string)
    df.to_csv(f"{EXPORT_DIR}/step8_tokenized.csv", index=False, encoding='utf-8')
    
    return df

def fit_tokenizer(texts):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def tokenize_and_pad(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences

def preprocess_text(text, tokenizer):
    text = lowercase_text(text)
    text = replace_lithuanian_characters_to_ascii(text, lithuanian_to_ascii_map)
    text = replace_slang(text, slang_dict)
    text = remove_special_characters(text)
    text = remove_stopwords(text, lithuanian_stopwords)
    text = ' '.join([remove_prefixes(word, lithuanian_prefixes) for word in text.split()])
    text = lithuanian_stemmer.stemWords(text.split())
    text = ' '.join(text)
    return tokenize_and_pad([text], tokenizer)

def load_tokenizer():
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer
