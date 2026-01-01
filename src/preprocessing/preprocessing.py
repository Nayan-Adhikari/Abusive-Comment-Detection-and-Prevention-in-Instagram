import re
import emoji
from src.lexicon.lexicon_loader import load_lexicon

NORMALIZATION_MAP, _, _ = load_lexicon()


def preprocess_comment(text: str):
    """
    Preprocesses a single Instagram comment.
    Returns tokens + intensity features.
    """

    features = {
        "emoji_count": 0,
        "repeat_intensity": 0,
        "exclaim_count": 0,
        "question_count": 0
    }

    # Lowercase
    text = text.lower().strip()

    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)

    # Emoji normalization
    demojized = emoji.demojize(text)
    tokens = demojized.split()
    cleaned_tokens = []

    for tok in tokens:
        if tok.startswith(":") and tok.endswith(":"):
            features["emoji_count"] += 1
            cleaned_tokens.append("<emoji>")
        else:
            cleaned_tokens.append(tok)

    text = " ".join(cleaned_tokens)

    # Repetition normalization
    def normalize_repeat(match):
        features["repeat_intensity"] += len(match.group()) - 2
        return match.group(1) * 2

    text = re.sub(r"(.)\1{2,}", normalize_repeat, text)

    # Punctuation intensity
    features["exclaim_count"] = text.count("!")
    features["question_count"] = text.count("?")

    text = re.sub(r"[^\w\s<>!?]", "", text)

    # Tokenization + dataset-driven normalization
    final_tokens = []
    for tok in text.split():
        final_tokens.append(NORMALIZATION_MAP.get(tok, tok))

    return {
        "tokens": final_tokens,
        "features": features
    }
