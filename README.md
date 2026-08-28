# 📩 Spam Message Classifier

A simple machine learning project that classifies SMS text messages as **Spam** or **Not Spam (Ham)** using TF-IDF and a Naive Bayes classifier.

## 🔍 Overview

This project demonstrates a complete, basic NLP + ML pipeline:
- Text cleaning and preprocessing
- Feature extraction with TF-IDF
- Classification with Multinomial Naive Bayes
- Evaluation using accuracy, precision, recall, and F1-score
- An interactive Streamlit demo app

## 🗂️ Project Structure

```
spam-classifier/
├── train.py           # Trains the model and saves it to disk
├── app.py              # Streamlit app to test the model interactively
├── requirements.txt    # Python dependencies
├── spam.csv             # Dataset (download separately, see below)
└── README.md
```

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset** (~5,500 labeled SMS messages).
Download `spam.csv` from Kaggle by searching "SMS Spam Collection Dataset" and place it in the project root.

## ⚙️ Setup

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Place `spam.csv` in the project folder.

3. Train the model:
```bash
python train.py
```
This prints evaluation metrics and saves `model.pkl` and `vectorizer.pkl`.

4. Run the interactive app:
```bash
streamlit run app.py
```

## 🧠 How It Works

1. **Text cleaning** — lowercase, strip punctuation/numbers, remove English stopwords.
2. **TF-IDF vectorization** — converts cleaned text into numerical features, weighting rare-but-informative words more heavily than common ones.
3. **Multinomial Naive Bayes** — a fast, strong baseline classifier well-suited to high-dimensional sparse text data.
4. **Evaluation** — since the dataset is imbalanced (far more "ham" than "spam"), accuracy alone is misleading, so precision, recall, and F1-score are tracked as well.

## 📈 Example Results

On a held-out 20% test split, this pipeline typically achieves ~96-98% accuracy, with strong precision on the spam class (few false alarms).

## 💬 Talking Points for Interviews

- **Why TF-IDF over raw word counts?** It downweights common words and upweights rare, distinctive ones, which tends to improve signal for classification.
- **Why Naive Bayes for text?** It handles high-dimensional, sparse data well, trains quickly, and is a strong baseline before trying more complex models.
- **Why precision/recall over accuracy?** With imbalanced classes, a model that always predicts "ham" would still score high accuracy — precision and recall reveal how well it actually catches spam without too many false positives.

## 🚀 Possible Extensions

- Try Logistic Regression or SVM and compare performance
- Add cross-validation instead of a single train/test split
- Deploy the Streamlit app publicly (Streamlit Community Cloud is free)
- Add a confusion matrix visualization to the app
