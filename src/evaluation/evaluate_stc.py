import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.pipeline.analyze_comment import analyze_comment

LABEL_MAP = {
    "SAFE": "SAFE",
    "WARNING": "WARNING",
    "SEVERE": "SEVERE",
    "CRITICAL": "SEVERE"
}

def evaluate_dataset(path, true_label):
    df = pd.read_excel(path)
    y_true = []
    y_pred = []

    for comment in df["raw_comment"].dropna().head(2000):  # limit for speed
        score, action = analyze_comment(comment)
        y_true.append(true_label)
        y_pred.append(LABEL_MAP[action])

    return y_true, y_pred


def run_evaluation():
    y_true, y_pred = [], []

    for path, label in [
        ("data/raw/safe_comments_100k.xlsx", "SAFE"),
        ("data/raw/warning_comments_100k.xlsx", "WARNING"),
        ("data/raw/severe_comments_100k.xlsx", "SEVERE")
    ]:
        t, p = evaluate_dataset(path, label)
        y_true.extend(t)
        y_pred.extend(p)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["SAFE", "WARNING", "SEVERE"]))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    run_evaluation()
