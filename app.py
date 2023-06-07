from flask import Flask, request, jsonify
from utils import embed_bert_cls, make_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the POST request
    text = data['text']  # Extract the input text from the JSON data

    # Call the make_prediction function
    prediction, probabilities = make_prediction(text)

    # Create a response JSON object with the prediction and probabilities
    response = {'prediction': prediction, 'probabilities': probabilities}

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run()
