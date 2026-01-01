import pandas as pd

LEXICON_PATH = "data/processed/final_processed_dataset_v2.xlsx"


def load_lexicon():
    """
    Loads the toxicity lexicon from Excel and builds lookup tables.

    Returns:
        normalization_map : raw_form -> normalized_form
        category_map      : normalized_form -> category
        weight_map        : normalized_form -> toxicity_weight
    """
    df = pd.read_excel(LEXICON_PATH)

    normalization_map = {}
    category_map = {}
    weight_map = {}

    for _, row in df.iterrows():
        raw = str(row["raw_comment"]).lower().strip()
        norm = str(row["normalized_form"]).lower().strip()

        normalization_map[raw] = norm
        category_map[norm] = row["category"]
        weight_map[norm] = float(row["toxicity_weight"])

    return normalization_map, category_map, weight_map
