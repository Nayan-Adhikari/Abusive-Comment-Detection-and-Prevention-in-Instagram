from src.preprocessing.preprocessing import preprocess_comment
from src.scoring.stc_scoring import compute_toxicity_score
from src.decision.decision_engine import decide_action


def analyze_comment(comment: str):
    preprocessed = preprocess_comment(comment)
    score = compute_toxicity_score(preprocessed)
    action = decide_action(score)
    return score, action
