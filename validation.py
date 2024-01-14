#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import load_model
from preprocessing import preprocess_data, load_tokenizer, tokenize_and_pad
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
from model import f1, precision, recall
import json

def load_history(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
MODEL_SAVE_PATH = './model/hate_speech_model.keras'
DATA_PATH = './datasets/lithuanian/train_tweets_lt.csv'  
TEST_SPLIT_SIZE = 0.4

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
test_data = preprocess_data(test_data, 'tweet')
X_test = test_data['tweet']
y_test = test_data['label']
tokenizer = load_tokenizer()
X_test_pad = tokenize_and_pad(X_test, tokenizer)

# Make predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.round(y_pred).astype(int).flatten()

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

#%%
# Print confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#%%
# Print classification report
print(classification_report(y_test, y_pred_classes))

#%%
# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, lw=2, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()