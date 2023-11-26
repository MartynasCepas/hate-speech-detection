from flask import Flask, request, jsonify
from keras.models import load_model
from preprocessing import preprocess_text, load_tokenizer
import traceback

app = Flask(__name__)
MODEL_PATH = './model/hate_speech_model.keras'
model = None 

model = load_model(MODEL_PATH)
tokenizer = load_tokenizer()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data['text']
        print(f"Original text: {text}")  # Log the original text

        preprocessed_text = preprocess_text(text, tokenizer)
        print(f"Preprocessed text: {preprocessed_text}")  # Log the preprocessed text
        
        prediction = model.predict(preprocessed_text)
        print(f"Prediction: {prediction}")  # Log the prediction
        
        result = 'Hate speech' if prediction[0][0] > 0.5 else 'Not hate speech'
        return jsonify(result=result)
    except Exception as e:
        # Log the error, which can be viewed in the Flask output
        print(traceback.format_exc())
        # Return an error message
        return jsonify(error=str(e)), 500
    
if __name__ == '__main__':
    app.run(debug=True)