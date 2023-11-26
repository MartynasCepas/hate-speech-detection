#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import load_model
from preprocessing import preprocess_data, load_tokenizer, tokenize_and_pad

#%%
# Constants
MODEL_SAVE_PATH = './model/hate_speech_model.keras'
DATA_PATH = './datasets/lithuanian/train_tweets_lt.csv'  
TEST_SPLIT_SIZE = 0.3

# Load the tokenizer
tokenizer = load_tokenizer()

# Load and preprocess the dataset
train_data = pd.read_csv(DATA_PATH)
X = train_data['tweet']
y = train_data['label']

# Here you use the same train_test_split with the same random_state to recreate the validation set
from sklearn.model_selection import train_test_split
_, X_val, _, y_val = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=0)

X_val_pad = tokenize_and_pad(X_val, tokenizer)

model = load_model(MODEL_SAVE_PATH)

y_pred = model.predict(X_val_pad)
y_pred_binary = (y_pred > 0.5).astype('int32')

#%%
# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(y_val, y_pred_binary, classes=['Not Hate Speech', 'Hate Speech'])

#%%
# Function to plot the ROC curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(y_val, y_pred)

#%%
# Function to print the classification report
def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

print_classification_report(y_val, y_pred_binary)
