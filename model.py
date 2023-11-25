from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.callbacks import EarlyStopping

# Constants
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 128
MAX_SEQUENCE_LENGTH = 100
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 2
PATIENCE = 2

def build_model():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train_pad, y_train, X_val_pad, y_val):
    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE)]
    model.fit(X_train_pad, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val_pad, y_val), callbacks=callbacks)
    return model