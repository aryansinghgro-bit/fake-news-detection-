"""Text preprocessing utilities for the fake news detection pipeline.

This module is used by both training and inference to guarantee that text is
normalized identically in both stages.
"""

from __future__ import annotations

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Words that carry no signal for fake-news classification but inflate the
# vocabulary. Kept intentionally small so domain-specific terms are preserved.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "is", "am", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "ought", "i", "me",
    "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "what", "which", "who", "whom",
    "this", "that", "these", "those",
}


def clean_text(text: str) -> str:
    """Normalize a single piece of text.

    Steps:
      1. Coerce to string and strip.
      2. Lowercase.
      3. Remove URLs and HTML tags.
      4. Strip non-alphabetic characters (keeps words, drops punctuation/numbers).
      5. Collapse repeated whitespace.
      6. Remove English stop words.

    Returns an empty string if the input is None or contains no usable text.
    """
    if not text:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return " ".join(tokens)


def combine_and_clean(title: str, body: str) -> str:
    """Combine a headline and article body, then clean the result.

    The headline is weighted by being prepended twice — titles are strong
    signals for fake-news detection and this modestly improves separability.
    """
    title_clean = clean_text(title)
    body_clean = clean_text(body)
    combined = f"{title_clean} {title_clean} {body_clean}".strip()
    return combined
