def make_prediction(text):
    text_embed = embed_bert_cls(text, model, tokenizer)
    pred = lr_model.predict_proba(np.reshape(text_embed, (1, -1)))
    res = label_encoder.inverse_transform(pred.argmax(axis=1))
    return res, pred[0]

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()