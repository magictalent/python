import joblib

# Load model and vectorizer
model = joblib.load("models/spam_detector_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    return "Spam" if prediction == 1 else "Ham"

# Test
print(predict_message("Congratulations! You've won a $1000 Walmart gift card"))
print(predict_message("Are we still meeting tomorrow?"))
