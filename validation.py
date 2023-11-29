#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import load_model
from preprocessing import preprocess_data, load_tokenizer, tokenize_and_pad
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
import json

def load_history(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
MODEL_SAVE_PATH = './model/hate_speech_model.keras'
DATA_PATH = './datasets/lithuanian/train_tweets_lt.csv'  
TEST_SPLIT_SIZE = 0.4

history = load_history('./model/history.json')

#%%
# Plot accuracy
plt.figure(figsize=(8, 4))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%%
# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Plot precision, recall, and F1-score
if 'precision' in history and 'recall' in history and 'f1' in history:
    plt.figure(figsize=(8, 4))
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.plot(history['f1'], label='F1-Score')
    plt.title('Precision, Recall, and F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.legend()
    plt.show()

#%%
# Access the final epoch's metrics
final_epoch = len(history['val_accuracy'])

# Access the final scores from the history
final_metrics = {
    'Accuracy': history['val_accuracy'][-1],
    'Validation Loss': history['val_loss'][-1],
    'Precision': history['precision'][-1] if 'precision' in history else 'N/A',
    'Recall': history['recall'][-1] if 'recall' in history else 'N/A',
    'F1-Score': history['f1'][-1] if 'f1' in history else 'N/A'
}

# Print the final scores in a table format
print(f"{'Metric':<25}{'Value':<15}")
print(f"{'-' * 40}")
for metric, value in final_metrics.items():
    print(f"{metric:<25}{value:<15.4f}" if isinstance(value, float) else f"{metric:<25}{value:<15}")