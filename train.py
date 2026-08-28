"""
Spam Message Classifier - Training Script
Trains a Naive Bayes model on the SMS Spam Collection dataset
and saves the trained model + vectorizer for reuse.
"""

import re
import pickle

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---- 1. Setup ----
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def main():
    # ---- 2. Load data ----
    # Download 'spam.csv' from: Kaggle - "SMS Spam Collection Dataset"
    # Place it in the same folder as this script.
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    print("Class distribution:")
    print(df["label"].value_counts())
    print()

    # ---- 3. Clean text ----
    df["clean_message"] = df["message"].apply(clean_text)

    # ---- 4. Vectorize ----
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["clean_message"]).toarray()
    y = df["label"].map({"ham": 0, "spam": 1})

    # ---- 5. Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 6. Train ----
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ---- 7. Evaluate ----
    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:   ", recall_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # ---- 8. Save model + vectorizer ----
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nSaved model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()
