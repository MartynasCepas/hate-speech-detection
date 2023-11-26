from flask import Flask, request, jsonify
from keras.models import load_model
from preprocessing import tokenize_and_pad
import traceback
from preprocessing import preprocess_text

app = Flask(__name__)
MODEL_PATH = './model/hate_speech_model.keras'
model = None 

def load_model_on_demand():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model_on_demand()
        data = request.get_json(force=True)
        text = data['text']
        preprocessed_text = preprocess_text(text)
        prediction = model.predict(preprocessed_text)
        print(prediction)
        result = 'Hate speech' if prediction[0][0] > 0.5 else 'Not hate speech'
        return jsonify(result=result)
    except Exception as e:
        # Log the error, which can be viewed in the Flask output
        print(traceback.format_exc())
        # Return an error message
        return jsonify(error=str(e)), 500