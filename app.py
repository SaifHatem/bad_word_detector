from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("bad_word_model.joblib")

@app.route('/')
def home():
    return "Bad Word Detector is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    prediction = model.predict([text])[0]
    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
