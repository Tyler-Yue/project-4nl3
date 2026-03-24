import pandas as pd
import numpy as np
import re
from pathlib import Path

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess_df(df):
    title = df["question_title"].fillna("").astype(str)
    content = df["question_content"].fillna("").astype(str)
    question = (title + " " + content).str.strip()
    return question.apply(normalize_text)

def tokenize_for_vectorizer(text):
    text = text.lower().strip()
    tokens = re.findall(r"\b\w+\b", text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

train_df = pd.read_csv("training_data.csv")
train_labels = pd.read_csv("training_label.csv")
test_df = pd.read_csv("testing_data.csv")

train_df["text"] = preprocess_df(train_df)
test_df["text"] = preprocess_df(test_df)

y_train = train_labels["class_index"].astype(int).values

count_vectorizer = CountVectorizer(
    tokenizer=tokenize_for_vectorizer,
    lowercase=False,
    max_features=10000,
)

X_train = count_vectorizer.fit_transform(train_df["text"].values)
X_test = count_vectorizer.transform(test_df["text"].values)

classes = np.unique(y_train)

n_test = X_test.shape[0]
rng = np.random.default_rng(1)
y_pred_random = rng.choice(classes, size=n_test)
pd.DataFrame({"class_label": y_pred_random}).to_csv("random_baseline.csv", index=False)

majority_class = Counter(y_train).most_common(1)[0][0]
y_pred_majority = np.full(n_test, majority_class)
pd.DataFrame({"class_label": y_pred_majority}).to_csv("majority_baseline.csv", index=False)

lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
pd.DataFrame({"class_label": y_pred_lr}).to_csv("logistic_regression_baseline.csv", index=False)

rf = RandomForestClassifier(n_estimators=50, max_depth=30, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
pd.DataFrame({"class_label": y_pred_rf}).to_csv("random_forest_baseline.csv", index=False)

print("Done! Created 4 baseline CSV files.")
