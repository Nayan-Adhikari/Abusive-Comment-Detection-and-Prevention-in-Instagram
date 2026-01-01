from src.config.settings import (
    SAFE_THRESHOLD,
    WARNING_THRESHOLD,
    SEVERE_THRESHOLD
)


def decide_action(score: float) -> str:
    if score <= SAFE_THRESHOLD:
        return "SAFE"
    elif score <= WARNING_THRESHOLD:
        return "WARNING"
    elif score <= SEVERE_THRESHOLD:
        return "SEVERE"
    else:
        return "CRITICAL"
