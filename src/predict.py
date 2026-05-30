def predict_news(
    text,
    model,
    vectorizer
):

    vec = vectorizer.transform(
        [text]
    )

    pred = model.predict(vec)

    return pred[0]
