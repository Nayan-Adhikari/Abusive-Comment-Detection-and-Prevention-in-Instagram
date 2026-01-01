import pandas as pd
from collections import Counter
from src.preprocessing.preprocessing import preprocess_comment

def validate_lexicon(dataset_path: str, label: str):
    """
    Prints most common tokens in a dataset after preprocessing.
    Used only for qualitative validation.
    """
    df = pd.read_excel(dataset_path)
    token_counter = Counter()

    for comment in df["comment"].dropna():
        result = preprocess_comment(comment)
        token_counter.update(result["tokens"])

    print(f"\nTop tokens in {label} dataset:")
    for token, count in token_counter.most_common(20):
        print(f"{token}: {count}")
