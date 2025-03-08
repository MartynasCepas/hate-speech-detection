#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import load_model
from preprocessing import preprocess_data, load_tokenizer, remove_stopwords, tokenize_and_pad
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
from model import f1, precision, recall
import json
from wordcloud import WordCloud
from collections import Counter
import itertools

def load_history(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
MODEL_SAVE_PATH = './model/hate_speech_model.keras'
DATA_PATH = './datasets/v2/reddit_data_annotated.csv'
TEST_SPLIT_SIZE = 0.4
STOPWORDS_PATH = './data/stopwords.txt'

with open(STOPWORDS_PATH, 'r', encoding='utf-8') as file:
    lithuanian_stopwords = file.read().splitlines()
history = load_history('./model/history.json')

# Load model
custom_objects = {
    'f1': f1,
    'precision': precision,
    'recall': recall
}
model = load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)

# Load and preprocess the test dataset
test_data = pd.read_csv(DATA_PATH)

hate_speech_series = test_data[test_data['is_hate_speech'] == 1]['body']
hate_speech_no_stopwords = [remove_stopwords(text, lithuanian_stopwords) for text in hate_speech_series]
joined_hate_speech = ' '.join(hate_speech_no_stopwords)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(joined_hate_speech)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Neapykantos kalbos žodžių debesis')
plt.show()

test_data = preprocess_data(test_data, 'body')
X_test = test_data['body']
y_test = test_data['is_hate_speech']
tokenizer = load_tokenizer()
X_test_pad = tokenize_and_pad(X_test, tokenizer)

# Make predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.round(y_pred).astype(int).flatten()

#%%
# Plot accuracy
plt.figure(figsize=(8, 4))
plt.plot(history['accuracy'], label='Mokymo tikslumas')
plt.plot(history['val_accuracy'], label='Validacijos tikslumas')
plt.title('Mokymo ir validacijos tikslumas pagal epochas')
plt.xlabel('Epochos')
plt.ylabel('Tikslumas')
plt.legend()
plt.show()

#%%
# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(history['loss'], label='Mokymo nuostoliai')
plt.plot(history['val_loss'], label='Validacijos nuostoliai')
plt.title('Mokymo ir validacijos nuostoliai pagal epochas')
plt.xlabel('Epochos')
plt.ylabel('Nuostoliai')
plt.legend()
plt.show()

#%%
# Plot precision, recall, and F1-score
if 'precision' in history and 'recall' in history and 'f1' in history:
    plt.figure(figsize=(8, 4))
    plt.plot(history['precision'], label='Tikslumas')
    plt.plot(history['recall'], label='Atšaukimas')
    plt.plot(history['f1'], label='F1-įvertis')
    plt.title('Tikslumo, atšaukimo ir F1-mato kitimas per epochas')
    plt.xlabel('Epochos')
    plt.ylabel('Reikšmės')
    plt.legend()
    plt.show()

#%%
# Class Distribution
plt.figure(figsize=(6, 6))
test_data['is_hate_speech'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.title('Klasių pasiskirstymas (Neapykantos kalba vs. Ne-neapykantos kalba)')
plt.ylabel('')
plt.show()

#%%
# Print confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Painiavos matrica')
plt.xlabel('Prognozuota')
plt.ylabel('Tikroji')
plt.show()

#%%
# Print classification report
print(classification_report(y_test, y_pred_classes))

#%%
# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC kreivė (plotas = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Klaidingai teigiamų rodiklis')
plt.ylabel('Teisingai teigiamų rodiklis')
plt.title('Darbinės charakteristikos kreivė (ROC)')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, lw=2, color='blue')
plt.xlabel('Atšaukimas')
plt.ylabel('Tikslumas')
plt.title('Tikslumo-atšaukimo kreivė')
plt.show()
