import joblib

# Load trained model & vectorizer
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

# Test predictions
print(predict_sentiment("I really love this product!"))
print(predict_sentiment("This movie was terrible."))
print(predict_sentiment("It was an average experience."))
