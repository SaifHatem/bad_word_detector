from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This allows your Flutter app to communicate with this API

print("Loading model...")
model = joblib.load('bad_word_model.joblib')
print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text in JSON format'}), 400
    
    text = data['text']
    print(f"\nReceived text: {text}")
    
    # Make prediction
    prediction = model.predict([text])[0]
    print(f"Prediction: {prediction}")
    
    return jsonify({'result': prediction})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working!'})

if __name__ == '__main__':
    print("\nAPI is running! Access these endpoints:")
    print(" - POST http://localhost:5000/predict")
    print(" - GET  http://localhost:5000/test")
    app.run(host='0.0.0.0', port=5000)