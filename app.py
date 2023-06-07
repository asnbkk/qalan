import joblib
import numpy as np
import torch
from flask import Flask, jsonify, request
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)

label_encoder = joblib.load("label_encoder.joblib")
lr_model = joblib.load("logistic_regression_model.pkl")

pret_tokenizer = AutoTokenizer.from_pretrained(
    "kz-transformers/kaz-roberta-conversational"
)
pret_model = AutoModel.from_pretrained("kz-transformers/kaz-roberta-conversational")


def make_prediction(text: str) -> tuple:
    """
    Make a prediction on the given text.

    Args:
        text (str): The input text.

    Returns:
        tuple: A tuple containing the predicted label and the prediction probabilities.
    """
    text_embed = embed_bert_cls(text, pret_model, pret_tokenizer)
    pred = lr_model.predict_proba(np.reshape(text_embed, (1, -1)))
    res = label_encoder.inverse_transform(pred.argmax(axis=1))
    return res, pred[0]


def embed_bert_cls(
    text: str, model: torch.nn.Module, tokenizer: AutoTokenizer
) -> np.ndarray:
    """
    Embed the text using BERT model.

    Args:
        text (str): The input text.
        model: The BERT model.
        tokenizer: The BERT tokenizer.

    Returns:
        np.ndarray: The embedded representation of the text.
    """
    tokenizer_ = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in tokenizer_.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


@app.route("/predict", methods=["POST"])
def predict() -> jsonify:
    """
    Handle POST request for prediction.

    Returns:
        jsonify: The JSON response containing the prediction and probabilities.
    """
    data = request.get_json()
    text = data["text"]

    prediction, probabilities = make_prediction(text)

    response = {
        "prediction": prediction[0],
        "probabilities": float(np.max(probabilities)),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run()
