"""
Spam Message Classifier - Streamlit App
Run with: streamlit run app.py
Requires model.pkl and vectorizer.pkl (created by running train.py first).
"""

import re
import pickle

import streamlit as st
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_message(msg, model, vectorizer):
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = proba[result]
    return ("Spam" if result == 1 else "Not Spam"), confidence


st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam Message Classifier")
st.write("Enter a text message below and the model will predict whether it's spam.")

model, vectorizer = load_model()

user_input = st.text_area("Message:", height=100)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        label, confidence = predict_message(user_input, model, vectorizer)
        if label == "Spam":
            st.error(f"🚨 {label} (confidence: {confidence:.2%})")
        else:
            st.success(f"✅ {label} (confidence: {confidence:.2%})")
