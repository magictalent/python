import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
data = pd.read_csv("../data/sentiment_dataset.csv")

X = data['text']
y = data['label']

# 2. Text preprocessing + vectorization
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 4. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save model + vectorizer
joblib.dump(model, "../models/sentiment_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
