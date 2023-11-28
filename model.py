from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras import backend as K

# Constants
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 100
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 4

def build_model():
    model = Sequential([
        Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(LSTM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT),
        # dropout to prevent overfitting
        Dropout(0.5),
        # dense to connect the previous output with current layer
        Dense(128, activation="relu"),
        # dropout to prevent overfitting
        Dropout(0.5),
        # this is output layer, with 3 class (0, 1, 2)
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1,precision, recall])
    return model

def train_model(model, X_train_pad, y_train, X_val_pad, y_val, save_path):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train)

    class_weights_dict = dict(enumerate(class_weights))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_pad, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_pad, y_val),
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    return model, history

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2*((precisions*recalls)/(precisions+recalls+K.epsilon()))