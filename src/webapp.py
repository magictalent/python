from flask import Flask, render_template, request
import joblib
import os

# Load model and vectorizer
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"))


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    if request.method == "POST":
        user_text = request.form["user_input"]
        text_vec = vectorizer.transform([user_text])
        sentiment = model.predict(text_vec)[0]
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=False)
