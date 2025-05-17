from flask import Flask, render_template, request, jsonify
import json
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load intents JSON
with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

# Prepare data
texts = []
labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Flask setup
app = Flask(__name__)

# Prediction function
def predict_intent(user_input):
    try:
        vec = vectorizer.transform([user_input.lower()])
        return model.predict(vec)[0]
    except:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]
    print("User message:", user_message)

    if not user_message.strip():
        return jsonify({"response": "Please enter a valid message."})

    if re.search(r"[^a-zA-Z0-9\s\?\.\!']", user_message):
        return jsonify({"response": "You're entering wrong characters. Please avoid special symbols."})

    intent = predict_intent(user_message)
    print("Predicted intent:", intent)

    if intent:
        for item in data["intents"]:
            if item["tag"] == intent:
                response = random.choice(item["responses"])
                print("Bot response:", response)
                return jsonify({"response": response})

    # If no match found or model fails
    print("Fallback used.")
    return jsonify({"response": "Sorry, I didn't understand that. Can you try rephrasing?"})

if __name__ == "__main__":
    app.run(debug=True)
