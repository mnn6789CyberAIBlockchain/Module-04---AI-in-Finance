# FinSight AI — Financial News Insight Application

A local educational prototype for the Module 04 finance project. It combines:
- **Supervised machine learning:** TF-IDF text features + Multinomial Naive Bayes.
- **Sentiment labels:** positive, negative, or neutral.
- **Local LLM explanation:** Ollama generates a short context brief and risk/uncertainty note.
- **Market overview:** S&P 500, QQQ, Dow Jones, Russell 2000, VIX, 10-year and 2-year Treasury yields, gold, WTI crude, and Bitcoin with compact recent-history charts.
- **Web UI:** Flask + HTML/CSS + Chart.js sparklines.

> **Important:** `sample_data.csv` contains small, hand-written teaching examples. Metrics are for demonstrating a workflow, not evidence of robust real-world financial sentiment performance. This is not financial advice and does not predict stock prices.

## 1. Requirements

- Python 3.10 or newer recommended.
- Ollama is optional for sentiment classification, but needed for the local LLM insight.
- If using Ollama, install/pull a model such as `llama3.2:1b`.

## 2. Open the project in PowerShell

Change directory to the folder where you extracted this project. Example:

```powershell
cd "E:\2372\FinSight_Finance_News_Assignment"
```

Use your actual extracted folder path if it is different.

## 3. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can instead run the project using `.venv\Scripts\python.exe` for each command, or allow scripts for this terminal session with:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## 4. Train the sentiment model

```powershell
python model_training.py
```

This reads `sample_data.csv`, cleans the text, splits the data into training and test sets, fits TF-IDF + Multinomial Naive Bayes, prints accuracy and a classification report, then creates:
- `sentiment_model.pkl`
- `vectorizer.pkl`

Run this command again if you change the dataset.

## 5. (Optional) Prepare the local LLM

If Ollama is installed, open a second PowerShell window and run:

```powershell
ollama pull llama3.2:1b
ollama run llama3.2:1b
```

If the model is already installed, the pull command simply checks/downloads what is needed. Ollama normally exposes its API at `http://localhost:11434`. The app calls `/api/chat`. You can close the interactive `ollama run` session after confirming the model works if the Ollama background service is running.

To use another Ollama model, set an environment variable before launching Flask:

```powershell
$env:OLLAMA_MODEL = "your-installed-model-name"
```

If Ollama is unavailable, the app still predicts sentiment and displays a helpful local-LLM notice.

## 6. Start the Flask app

```powershell
python app.py
```

Open this URL in your browser:

`http://127.0.0.1:5000`

Stop the server with `Ctrl+C`.

## 7. Market dashboard data notes

The dashboard fetches index/ETF/commodity/Bitcoin data through `yfinance` and Treasury yields from FRED public CSV observations. These services require internet access; quotes may be delayed or temporarily unavailable. Yahoo Finance via `yfinance` is an unofficial access method and may change or rate-limit requests. The dashboard isolates data-source errors so the news-analysis form can still load. Treasury yield changes are shown in percentage points, not percent returns. Chart.js is loaded from a CDN, so trend charts require internet access in the browser.

## 8. Test cases

Try these separately:
1. Positive: `The company raised its revenue outlook after stronger-than-expected sales and improved margins.`
2. Negative: `The firm cut its forecast after falling demand and a sharp decline in quarterly profit.`
3. Neutral: `The company will release its quarterly results next Tuesday after the market closes.`

Because this is a small demo dataset, the classifier may occasionally choose an unexpected label. That is a useful limitation to mention in your presentation.

## Project structure

```text
FinSight_Finance_News_Assignment/
├── app.py
├── model_training.py
├── preprocess.py
├── market_data.py
├── news_feed.py
├── model_evaluation.txt
├── test_sentiment.py
├── sample_data.csv
├── requirements.txt
├── README.md
├── sentiment_model.pkl       # created by model_training.py
├── vectorizer.pkl            # created by model_training.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Explain it in a presentation

1. The user submits financial news in the Flask form.
2. `preprocess.py` normalizes the text.
3. The saved TF-IDF vectorizer converts the text into numerical features.
4. The trained Multinomial Naive Bayes classifier predicts sentiment.
5. The Flask app sends the news and predicted label to the local Ollama model for a concise explanation.
6. The page displays sentiment, model class scores, and the LLM context brief.

**Limitations:** tiny synthetic dataset; sentiment can be ambiguous; class scores are model probabilities, not calibrated confidence; LLM text can be incomplete or wrong; no trading signals or investment recommendations.
