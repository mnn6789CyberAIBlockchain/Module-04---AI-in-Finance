"""Flask web application for finance-news sentiment and local LLM insights."""
from pathlib import Path
import os
import requests
import joblib
from flask import Flask, render_template, request
from preprocess import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"

app = Flask(__name__)

def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python model_training.py` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

def get_local_llm_insight(news_text: str, sentiment: str) -> str:
    """Ask Ollama for a cautious summary. The app remains usable if Ollama is offline."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    prompt = (
        "You are a cautious financial-news explainer, not a financial adviser. "
        "Use only the provided text. In 2-4 concise sentences, summarize the main event, "
        "explain why it may matter to investors, and mention one uncertainty or risk. "
        "Do not invent facts, prices, dates, or recommendations. Do not tell the user to buy or sell.\n\n"
        f"Classifier sentiment: {sentiment}\nNews text: {news_text}"
    )
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model_name,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "Be concise, grounded, and cautious."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        insight = data.get("message", {}).get("content", "").strip()
        return insight or "Ollama responded, but no insight text was returned."
    except requests.exceptions.ConnectionError:
        return (
            "Local LLM unavailable: Ollama could not be reached. Start Ollama and ensure "
            f"the model '{model_name}' is installed. Sentiment classification still works."
        )
    except requests.exceptions.Timeout:
        return "The local LLM took too long to respond. Try again or use a smaller Ollama model."
    except requests.exceptions.RequestException as exc:
        return f"Could not get a local LLM insight ({exc.__class__.__name__}). Check Ollama is running."
    except (ValueError, TypeError):
        return "Ollama returned an unexpected response. Check the local model/API configuration."

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    news_text = ""
    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if not news_text:
            error = "Paste a finance-related headline or news paragraph first."
        elif len(news_text) > 8000:
            error = "Please keep the input under 8,000 characters."
        else:
            try:
                model, vectorizer = load_artifacts()
                cleaned = clean_text(news_text)
                vector = vectorizer.transform([cleaned])
                sentiment = str(model.predict(vector)[0]).title()
                probabilities = {}
                if hasattr(model, "predict_proba"):
                    for label, probability in zip(model.classes_, model.predict_proba(vector)[0]):
                        probabilities[str(label).title()] = round(float(probability) * 100, 1)
                result = {
                    "sentiment": sentiment,
                    "probabilities": probabilities,
                    "insight": get_local_llm_insight(news_text, sentiment),
                }
            except FileNotFoundError as exc:
                error = str(exc)
            except Exception:
                error = "The app could not process this input. Check the dataset/model setup and try again."
    return render_template("index.html", result=result, error=error, news_text=news_text)

if __name__ == "__main__":
    print("FinSight AI is starting at http://127.0.0.1:5000")
    app.run(debug=True)
