
"""FinSight AI: finance-news sentiment, market dashboard, and local LLM insights."""

from pathlib import Path
import os

import joblib
import requests
from flask import Flask, render_template, request

from preprocess import clean_text
from market_data import get_market_dashboard
from news_feed import fetch_financial_news

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"

app = Flask(__name__)


# --------------------------------------------------
# 2. LOAD THE TRAINED MACHINE-LEARNING MODEL
# --------------------------------------------------

def load_artifacts():
    """Load the trained sentiment classifier and TF-IDF vectorizer."""

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python model_training.py` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# --------------------------------------------------
# 3. GENERATE AN INSIGHT USING LOCAL OLLAMA
# --------------------------------------------------

def get_local_llm_insight(news_text: str, sentiment: str) -> str:
    """
    Ask a local Ollama model to explain financial news.

    Sentiment classification continues to work even when Ollama
    is unavailable.
    """

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    endpoint = os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434/api/chat"
    )

    prompt = (
        "You are a cautious financial-news explainer, not a financial adviser. "
        "Use only the provided text. In 2-4 concise sentences, summarize the "
        "main event, explain why it may matter to investors, and mention one "
        "uncertainty or risk. Do not invent facts, prices, dates, or "
        "recommendations. Do not tell the user to buy or sell.\n\n"
        f"Classifier sentiment: {sentiment}\n"
        f"News text: {news_text}"
    )

    try:
        response = requests.post(
            endpoint,
            json={
                "model": model_name,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Be concise, grounded, and cautious. "
                            "Do not invent financial information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            },
            timeout=45,
        )

        response.raise_for_status()

        data = response.json()
        insight = data.get("message", {}).get("content", "").strip()

        if insight:
            return insight

        return "Ollama responded, but no insight text was returned."

    except requests.exceptions.ConnectionError:
        return (
            "Local LLM unavailable: Ollama could not be reached. "
            f"Start Ollama and ensure '{model_name}' is installed. "
            "Sentiment classification still works."
        )

    except requests.exceptions.Timeout:
        return (
            "The local LLM took too long to respond. "
            "Try again or use a smaller Ollama model."
        )

    except requests.exceptions.RequestException as exc:
        return (
            f"Could not get a local LLM insight ({exc.__class__.__name__}). "
            "Check that Ollama is running."
        )

    except (ValueError, TypeError):
        return (
            "Ollama returned an unexpected response. "
            "Check the local model and API configuration."
        )


# --------------------------------------------------
# 4. MARKET DASHBOARD DATA
# --------------------------------------------------

def load_market_data():
    """
    Retrieve all market cards from market_data.py.

    Return a fallback structure if the dashboard function itself fails,
    so a market-data problem does not prevent the news app from loading.
    """

    try:
        market = get_market_dashboard()

        if not isinstance(market, dict):
            raise ValueError(
                "get_market_dashboard() did not return a dictionary."
            )

        if "cards" not in market:
            raise ValueError(
                "The market dashboard response is missing its 'cards' list."
            )

        return market

    except Exception as exc:
        app.logger.exception("Unable to load the market dashboard.")

        return {
            "generated_at": "Currently unavailable",
            "cards": [],
            "error": (
                "Market data could not be loaded. "
                f"Check market_data.py. Error: {exc.__class__.__name__}"
            ),
        }


# --------------------------------------------------
# 5. MAIN PAGE: NEWS ANALYSIS + MARKET OVERVIEW
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    news_text = ""

    # Process the submitted financial-news text.
    if request.method == "POST":

        news_text = request.form.get("news_text", "").strip()

        if not news_text:
            error = (
                "Paste a finance-related headline or news paragraph first."
            )

        elif len(news_text) > 8000:
            error = "Please keep the input under 8,000 characters."

        else:
            try:
                # Load the trained classifier and vectorizer.
                model, vectorizer = load_artifacts()

                # Clean and transform the submitted text.
                cleaned = clean_text(news_text)
                vector = vectorizer.transform([cleaned])

                # Predict the news sentiment.
                sentiment = str(model.predict(vector)[0]).title()

                # Calculate model probability scores, when supported.
                probabilities = {}

                if hasattr(model, "predict_proba"):
                    predicted_probabilities = model.predict_proba(vector)[0]

                    for label, probability in zip(
                        model.classes_,
                        predicted_probabilities,
                    ):
                        probabilities[str(label).title()] = round(
                            float(probability) * 100,
                            1,
                        )

                # Ask Ollama for an explanation of the submitted news.
                insight = get_local_llm_insight(
                    news_text,
                    sentiment,
                )

                # Pass analysis results to index.html.
                result = {
                    "sentiment": sentiment,
                    "probabilities": probabilities,
                    "insight": insight,
                }

            except FileNotFoundError as exc:
                error = str(exc)

            except Exception:
                app.logger.exception("News analysis failed.")

                error = (
                    "The app could not process this input. "
                    "Check the dataset, trained model, and dependencies, "
                    "then try again."
                )

    # Retrieve the market dashboard independently of news analysis.
    # Load market data and recent financial news.
    market = load_market_data()
    news_items = fetch_financial_news(limit=15)

    # Render the dashboard.
    return render_template(
        "index.html",
        result=result,
        error=error,
        news_text=news_text,
        market=market,
        news_items=news_items,
    )


if __name__ == "__main__":
    print("=" * 55)
    print("FinSight AI is starting at http://127.0.0.1:5000")
    app.run(debug=True)

# --------------------------------------------------
# 6. START THE FLASK APPLICATION
# --------------------------------------------------


# --------------------------------------------------
# 6. START THE FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("FinSight AI - Finance News Insight")
    print("Local website: http://127.0.0.1:5000")
    print("Market dashboard: Enabled")
    print("News sentiment: TF-IDF + Multinomial Naive Bayes")
    print("Local LLM: Ollama")
    print("=" * 55)

    app.run(debug=True)

