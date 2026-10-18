
"""Fetch and filter recent financial news from RSS feeds."""

import calendar
import html
import re
import time
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests


NEWS_FEEDS = [
    {
        "name": "CNBC Markets",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20409666",
    },
    {
        "name": "CNBC Business",
        "url": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    },
    {
        "name": "Reuters Markets",
        "url": (
            "https://news.google.com/rss/search?"
            "q=when:7d+site:reuters.com/markets"
            "&ceid=US:en&hl=en-US&gl=US"
        ),
    },
]

# These phrases usually indicate quote pages or product listings,
# rather than substantive news articles.
LOW_VALUE_PATTERNS = [
    r"\bstock price\s*(?:&|and)?\s*latest news\b",
    r"\bstock price and news\b",
    r"\bstock price\s*\|\s*",
    r"\bstock quote\b",
    r"\bquote and news\b",
    r"^\s*about .+\b(?:ETF|exchange-traded fund)\b",
    r"\b(?:ETF|exchange-traded fund)\b.{0,60}\bstock price\b",
]

# Avoid treating a ticker-only page as a news story.
TICKER_ONLY_PATTERN = re.compile(
    r"^\s*\([A-Z0-9.^-]{1,15}\)\s*\|\s*"
    r"(?:stock price|latest news|quotes?)\b",
    re.IGNORECASE,
)


def clean_summary(text):
    """Remove HTML tags and normalize RSS text."""

    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title):
    """Normalize titles for duplicate detection."""

    title = html.unescape(title or "").lower()
    title = re.sub(r"\s*[-|]\s*(?:reuters|cnbc)\s*$", "", title)
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def is_low_value_story(title, summary):
    """Reject obvious quote pages and low-value product listings."""

    text = f"{title} {summary}".lower()

    if TICKER_ONLY_PATTERN.search(title):
        return True

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in LOW_VALUE_PATTERNS
    )


def normalize_link(link):
    """Normalize URLs to reduce duplicate links."""

    try:
        parts = urlsplit(link.strip())
        # Remove query parameters and fragments, which often track clicks.
        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            "",
        ))
    except ValueError:
        return link.strip()


def fetch_financial_news(limit=15):
    """Fetch recent stories, filter duplicates, and handle feed errors."""

    stories = []
    seen_links = set()
    seen_titles = set()

    for source in NEWS_FEEDS:
        try:
            response = requests.get(
                source["url"],
                headers={"User-Agent": "FinSightAI/1.0 news reader"},
                timeout=12,
            )
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            if feed.bozo and not feed.entries:
                print(f"News feed warning ({source['name']}): invalid feed")
                continue

            for entry in feed.entries:
                title = clean_summary(entry.get("title", ""))
                link = entry.get("link", "").strip()

                summary = clean_summary(
                    entry.get("summary", entry.get("description", ""))
                )

                if not title or not link:
                    continue

                if is_low_value_story(title, summary):
                    continue

                normalized_link = normalize_link(link)
                normalized_title = normalize_title(title)

                if (
                    not normalized_link
                    or normalized_link in seen_links
                    or not normalized_title
                    or normalized_title in seen_titles
                ):
                    continue

                published = entry.get("published_parsed")

                if published:
                    published_at = time.strftime(
                        "%b %d, %Y %H:%M UTC",
                        published,
                    )
                    published_sort = calendar.timegm(published)
                else:
                    published_at = "Publication time unavailable"
                    published_sort = 0

                stories.append({
                    "title": title,
                    "summary": summary[:700],
                    "link": link,
                    "source": source["name"],
                    "published": published_at,
                    "published_sort": published_sort,
                })

                seen_links.add(normalized_link)
                seen_titles.add(normalized_title)

        except requests.RequestException as exc:
            print(
                f"News feed warning ({source['name']}): "
                f"{exc.__class__.__name__}"
            )

    stories.sort(
        key=lambda story: story["published_sort"],
        reverse=True,
    )

    return stories[:max(0, limit)]
