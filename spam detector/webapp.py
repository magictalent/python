from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("models/spam_detector_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        message = request.form["message"]
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)[0]
        result = "Spam 🚨" if prediction == 1 else "Ham ✅"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False)
