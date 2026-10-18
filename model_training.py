
"""Train and evaluate a TF-IDF + Multinomial Naive Bayes finance sentiment classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text


# --------------------------------------------------
# 1. FILE PATHS AND VALID SENTIMENT LABELS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "sample_data.csv"
MODEL_PATH = ROOT / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"
REPORT_PATH = ROOT / "model_evaluation.txt"

VALID_LABELS = {"positive", "negative", "neutral"}
LABEL_ORDER = sorted(VALID_LABELS)


# --------------------------------------------------
# 2. LOAD AND PREPROCESS THE DATASET
# --------------------------------------------------

def load_and_preprocess_data(path=DATA_PATH):
    """Load, validate, and clean the finance sentiment dataset."""

    df = pd.read_csv(path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "sample_data.csv must contain 'text' and 'label' columns."
        )

    # Remove rows with missing text or labels.
    df = df.dropna(subset=["text", "label"]).copy()

    # Clean the text and standardize sentiment labels.
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Report duplicate cleaned text before filtering.
    duplicate_count = int(df["text"].duplicated().sum())
    print(f"Duplicate cleaned text rows found: {duplicate_count}")

    # Remove empty text and invalid labels.
    df = df[
        (df["text"].str.len() > 0)
        & df["label"].isin(VALID_LABELS)
    ].copy()

    # Validate the dataset before splitting.
    if df["label"].nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two valid sentiment labels."
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Each sentiment label needs at least two rows "
            "for a stratified train/test split."
        )

    print("\nSentiment label distribution:")
    print(df["label"].value_counts().sort_index())

    return df


# --------------------------------------------------
# 3. TRAIN AND EVALUATE THE MODEL
# --------------------------------------------------

def train_sentiment_model(path=DATA_PATH):
    """Train the classifier, evaluate performance, and save artifacts."""

    df = load_and_preprocess_data(path)

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    # Convert text into numerical TF-IDF features.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )

    # Fit the vectorizer on training data only to prevent data leakage.
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize and train the Naive Bayes classifier.
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    # Generate predictions for the unseen test set.
    predictions = model.predict(X_test_vec)

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABEL_ORDER,
    )

    # Save the trained model and vectorizer.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # --------------------------------------------------
    # 4. SAVE THE EVALUATION REPORT
    # --------------------------------------------------

    label_distribution = df["label"].value_counts().sort_index().to_dict()

    report_text = (
        "FinSight AI - Sentiment Model Evaluation\n"
        "========================================\n\n"
        f"Dataset rows used: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Label distribution: {label_distribution}\n"
        f"TF-IDF features: {len(vectorizer.get_feature_names_out())}\n"
        f"Naive Bayes alpha: {model.alpha}\n"
        f"Test accuracy: {accuracy:.4f}\n"
        f"Macro F1-score: {macro_f1:.4f}\n\n"
        "Classification Report:\n"
        f"{report}\n"
        f"Confusion Matrix (labels: {LABEL_ORDER}):\n"
        f"{matrix}\n\n"
        "Limitations:\n"
        "This evaluation uses a single stratified train/test split.\n"
        "Performance depends on the quality and size of the dataset.\n"
        "Results may not generalize to real-world financial news.\n"
        "Sentiment classification is not a prediction of market returns.\n"
    )

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    # --------------------------------------------------
    # 5. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 62)
    print("FinSight AI - Finance News Sentiment Model")
    print("=" * 62)

    print(f"Dataset rows used: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"TF-IDF features: {len(vectorizer.get_feature_names_out())}")

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification Report:\n")
    print(report)

    print(f"Confusion Matrix (labels: {LABEL_ORDER}):")
    print(matrix)

    print("\nSaved files:")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Vectorizer: {VECTORIZER_PATH.name}")
    print(f"Evaluation report: {REPORT_PATH.name}")

    print("\nImportant limitation:")
    print(
        "sample_data.csv is a teaching/demo dataset. "
        "These metrics do not establish real-world financial prediction "
        "performance."
    )

    return model, vectorizer, accuracy, report


# --------------------------------------------------
# 6. RUN THE TRAINING SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    train_sentiment_model()
