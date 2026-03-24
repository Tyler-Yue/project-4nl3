import pandas as pd
import numpy as np
import re
from pathlib import Path

import nltk
# Download NLTK data once (stopwords, wordnet for lemmatization)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# --- Data paths ---
DATA_DIR = Path(".")
TRAIN_DATA_PATH = DATA_DIR / "training_data.csv"
TRAIN_LABEL_PATH = DATA_DIR / "training_label.csv"
TEST_DATA_PATH = DATA_DIR / "testing_data.csv"

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


# Load train and test data
train_text_df = pd.read_csv(TRAIN_DATA_PATH)
train_label_df = pd.read_csv(TRAIN_LABEL_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# Merge train labels into train data
train_df = train_text_df.copy()
train_df["class_index"] = pd.to_numeric(train_label_df["class_index"], errors="coerce").astype("Int64")

# Preprocess text and store in new column
train_df["text"] = preprocess_df(train_df)
test_df["text"] = preprocess_df(test_df)

# Target: class_index
train_df["class_index"] = pd.to_numeric(train_df["class_index"], errors="coerce").astype("Int64")

# Clean: drop rows with missing class
train_df = train_df.dropna(subset=["class_index"])

# Use all training data for fitting; testing_data order is preserved for outputs
X_train_raw = train_df["text"].values
y_train = train_df["class_index"].astype(int).values
X_test_raw = test_df["text"].values

print("Train size:", len(y_train), "Test size:", len(X_test_raw))
print("Class distribution (train):\n", pd.Series(y_train).value_counts().sort_index())
print("----------")
print("Sample cleaned text (train):")
print(X_train_raw[:3])

# Vectorize text using CountVectorizer on train, then transform test
count_vectorizer = CountVectorizer(
    tokenizer=tokenize_for_vectorizer,
    lowercase=False,
    max_features=10000,
)

X_train = count_vectorizer.fit_transform(X_train_raw)
X_test = count_vectorizer.transform(X_test_raw)
print("Feature matrix shapes - train:", X_train.shape, "test:", X_test.shape)


def save_predictions_csv(file_name, predictions):
    out_df = pd.DataFrame({"class_label": np.asarray(predictions, dtype=int)})
    out_df.to_csv(DATA_DIR / file_name, index=False)
    print(f"Saved predictions: {file_name}")

# Random Baseline Model
rng = np.random.default_rng(1)
classes = np.unique(y_train)
y_pred_random = rng.choice(classes, size=len(X_test_raw))
save_predictions_csv("random_baseline.csv", y_pred_random)

# Majority Vote Baseline Model
from collections import Counter
majority_class = Counter(y_train).most_common(1)[0][0]
y_pred_majority = np.full(len(X_test_raw), majority_class)
save_predictions_csv("majority_baseline.csv", y_pred_majority)

# Logistical Regression Baseline Model 
lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
save_predictions_csv("logistic_regression_baseline.csv", y_pred_lr)

# Random Forest Baseline Model
rf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
save_predictions_csv("random_forest_baseline.csv", y_pred_rf)