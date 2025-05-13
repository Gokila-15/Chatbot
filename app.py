from flask import Flask, render_template, request, jsonify
import json
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load intents from the JSON file
with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

# Prepare the dataset
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize the patterns
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Flask app setup
app = Flask(_name_)

# Predict the intent of the user input
def predict_intent(user_input):
    if not user_input.strip():
        return None
    try:
        input_tfidf = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_tfidf)[0]
        return prediction
    except:
        return None

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for handling chat requests
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]

    # Check for invalid input
    if not user_message.strip():
        return jsonify({"response": "Please enter a valid message."})

    if re.search(r"[^a-zA-Z0-9\s\?\.\!']", user_message):
        return jsonify({"response": "You're entering wrong characters. Please avoid special symbols."})

    # Predict intent
    intent_tag = predict_intent(user_message)

    # Return matching response
    if intent_tag:
        for intent in data["intents"]:
            if intent["tag"] == intent_tag:
                response = random.choice(intent["responses"])
                break
        else:
            response = "Sorry, I couldn't find a suitable response."
    else:
        response = "I'm not sure how to respond to that. Try rephrasing your message."

    return jsonify({"response": response})

# Run the app
if _name_ == "_main_":
    app.run(debug=True, host="0.0.0.0", port=5000)
