
"""Test FinSight AI using financial headlines not in the training CSV."""

from pathlib import Path

import joblib

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent

model = joblib.load(ROOT / "sentiment_model.pkl")
vectorizer = joblib.load(ROOT / "vectorizer.pkl")

test_headlines = [
    "Shares surged after the company raised its earnings forecast.",
    "The company cut its outlook after quarterly revenue fell sharply.",
    "The board scheduled its annual shareholder meeting for next month.",
    "Revenue beat expectations, but management warned that demand may weaken.",
    "The company reported higher profits despite a decline in total sales.",
    "The firm announced a new chief financial officer effective Monday.",
    "Rising costs forced the retailer to lower its profit guidance.",
    "The company maintained its full-year outlook following mixed results.",
    "The bank reported increasing loan defaults and weaker credit quality.",
]

print("=" * 65)
print("FinSight AI - Unseen Headline Tests")
print("=" * 65)

for headline in test_headlines:
    cleaned = clean_text(headline)
    features = vectorizer.transform([cleaned])

    prediction = str(model.predict(features)[0]).title()

    print(f"\nHeadline: {headline}")
    print(f"Predicted sentiment: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        scores = {
            str(label).title(): round(float(probability) * 100, 1)
            for label, probability in zip(model.classes_, probabilities)
        }
        print(f"Class probabilities: {scores}")
