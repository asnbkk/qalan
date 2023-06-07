from flask import Flask, request, jsonify

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import joblib


app = Flask(__name__)

label_encoder = joblib.load("label_encoder.joblib")
lr_model = joblib.load("logistic_regression_model.pkl")

tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")
model = AutoModel.from_pretrained("kz-transformers/kaz-roberta-conversational")


def make_prediction(text):
    text_embed = embed_bert_cls(text, model, tokenizer)
    pred = lr_model.predict_proba(np.reshape(text_embed, (1, -1)))
    res = label_encoder.inverse_transform(pred.argmax(axis=1))
    return res, pred[0]


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get the JSON data from the POST request
    text = data["text"]  # Extract the input text from the JSON data

    # Call the make_prediction function
    prediction, probabilities = make_prediction(text)

    # Create a response JSON object with the prediction and probabilities
    response = {
        "prediction": prediction[0],
        "probabilities": float(np.max(probabilities)),
    }

    # Return the response as JSON
    return jsonify(response)


if __name__ == "__main__":
    app.run()
