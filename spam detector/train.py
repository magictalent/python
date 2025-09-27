import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]   # keep only label and message
data.columns = ['label', 'message']

# 2. Convert labels (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham':0, 'spam':1})

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 4. Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Test accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Save model and vectorizer
joblib.dump(model, "models/spam_detector_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
