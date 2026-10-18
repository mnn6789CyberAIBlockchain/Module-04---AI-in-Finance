"""Shared text preprocessing utilities for the FinSight finance sentiment project."""
import re
import unicodedata

def clean_text(text: str) -> str:
    """Normalize text while retaining finance terms, numbers, and negation."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9$%.,!?'\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
