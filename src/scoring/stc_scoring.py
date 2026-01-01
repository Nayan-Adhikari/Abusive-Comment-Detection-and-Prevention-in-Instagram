from src.lexicon.lexicon_loader import load_lexicon
from src.config.settings import (
    EMOJI_WEIGHT,
    REPEAT_WEIGHT,
    PUNCTUATION_WEIGHT
)

_, _, WEIGHT_MAP = load_lexicon()


def compute_toxicity_score(preprocessed: dict) -> float:
    """
    Computes toxicity score using rule-based aggregation.
    """

    tokens = preprocessed["tokens"]
    features = preprocessed["features"]

    base_score = 0.0
    for tok in tokens:
        if tok in WEIGHT_MAP:
            base_score = max(base_score, WEIGHT_MAP[tok])

    emoji_boost = features["emoji_count"] * EMOJI_WEIGHT
    repeat_boost = features["repeat_intensity"] * REPEAT_WEIGHT
    punct_boost = (
        features["exclaim_count"] + features["question_count"]
    ) * PUNCTUATION_WEIGHT

    score = base_score + emoji_boost + repeat_boost + punct_boost
    return min(1.0, score)
