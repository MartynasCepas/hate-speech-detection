from flask import Flask, request, jsonify
from keras.models import load_model
from preprocessing import tokenize_and_pad

app = Flask(__name__)
MODEL_PATH = './model/hate_speech_model.h5'
model = None 

def load_model_on_demand():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    processed_text = tokenize_and_pad([text])
    prediction = model.predict(processed_text)
    result = 'Hate speech' if prediction[0][0] > 0.5 else 'Not hate speech'
    return jsonify(result=result)