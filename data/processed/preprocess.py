
"""
Preprocessing script for Abusive Comment Detection and Prevention (Hinglish).
Saves cleaned output to data/processed/hinglish_clean.csv by default.

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --input data/raw/hinglish_comments_15000.csv --output data/processed/hinglish_clean.csv
"""

import argparse
import os
import re
import sys
import csv
from pathlib import Path
from typing import List
from tqdm import tqdm

import pandas as pd


try:
    import emoji
except Exception:
    emoji = None


USE_NLTK = True
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:
    USE_NLTK = False
    nltk = None
    nltk_stopwords = None

# ---------------------
# Configuration
# ---------------------
DEFAULT_INPUT = "data/raw/hinglish_comments_15000.csv"
DEFAULT_OUTPUT = "data/processed/hinglish_clean.csv"
SAMPLE_ROWS_TO_PRINT = 8

# A small Hinglish normalization map (expand as needed)
HINGLISH_MAP = {
    "bhaiyya": "bhai",
    "bhaiya": "bhai",
    "bhaiyy": "bhai",
    "yaar": "yaar",
    "yaarr": "yaar",
    "bahot": "bahut",
    "bhot": "bahut",
    "bcz": "kyunki",
    "kya": "kya",
    "tum": "tum",
    "u": "tum",
    "tu": "tu",
    "sahi": "sahi",
    "acha": "accha",
    "accha": "accha",
    "acha?": "accha",
    "thk": "theek",
    "theek": "theek",
    "plz": "please",
    "pls": "please",
    "bc": "bitch",  
}

PUNCT_KEEP = "?!"

# ---------------------
# Helpers
# ---------------------
def ensure_dirs_for_file(path: str):
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    # read with engine and low_memory off
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    return df

def replace_emoji(text: str) -> str:
    """Remove or replace emojis. Uses emoji lib if available, otherwise naive regex."""
    if text is None:
        return ""
    if emoji:
        try:
            return emoji.replace_emoji(text, replace="")
        except Exception:
            pass
    # fallback regex (approx) to strip common unicode emoji ranges
    try:
        # remove most non-word symbols and emoticons
        return re.sub(r"[\U00010000-\U0010ffff]", "", text)
    except re.error:
        return text

def remove_urls_mentions_hashtags(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    # remove hashtags but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)
    return text

def normalize_repeated_chars(text: str) -> str:
    # Replace 3+ repeated letters with 2 (cooool -> coool -> cool-ish)
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def remove_special_chars_keep_basic(text: str) -> str:
    # Keep letters, numbers, spaces and ? !
    keep = PUNCT_KEEP
    pattern = rf"[^a-zA-Z0-9\s{re.escape(keep)}]"
    text = re.sub(pattern, " ", text)
    return text

def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def apply_hinglish_map(text: str) -> str:
    # Replace tokens based on HINGLISH_MAP
    def repl_word(w):
        lw = w.lower()
        return HINGLISH_MAP.get(lw, w)
    tokens = text.split()
    tokens = [repl_word(t) for t in tokens]
    return " ".join(tokens)

def lowercase(text: str) -> str:
    return text.lower()

def remove_stopwords(text: str, stopwords_set: set) -> str:
    tokens = text.split()
    filtered = [t for t in tokens if t not in stopwords_set]
    return " ".join(filtered)

# ---------------------
# Main cleaning pipeline
# ---------------------
def clean_text(raw: str, stopwords_set: set = None, remove_stopwords_flag: bool = False) -> str:
    if raw is None:
        return ""
    text = str(raw)
    text = text.strip()
    text = lowercase(text)
    text = replace_emoji(text)
    text = remove_urls_mentions_hashtags(text)
    text = normalize_repeated_chars(text)
    text = apply_hinglish_map(text)
    text = remove_special_chars_keep_basic(text)
    text = collapse_spaces(text)
    if remove_stopwords_flag and stopwords_set:
        text = remove_stopwords(text, stopwords_set)
    return text

# ---------------------
# NLTK stopwords helper
# ---------------------
def get_stopwords() -> set:
    # returns a small combined English + Hindi stopword set (if nltk available)
    base = set()
    # basic common english stopwords fallback
    fallback = {
        "the", "is", "in", "at", "which", "on", "and", "a", "an", "to", "for", "of", "this",
        "that", "it", "i", "you", "he", "she", "we", "they", "are", "was", "with", "but",
        "or", "as", "by", "from", "be", "has", "have", "do", "does", "did", "not"
    }
    base.update(fallback)
    # small common Hinglish/Hindi function words to remove if desired
    hindi_like = {"hai", "hai?", "ka", "ke", "ki", "ko", "mein", "me", "se", "kaise", "kya", "kuch", "par", "aur"}
    base.update(hindi_like)
    if USE_NLTK:
        try:
            nltk.data.find("corpora/stopwords")
        except Exception:
            try:
                nltk.download("stopwords")
            except Exception:
                pass
        try:
            en = set(nltk_stopwords.words("english"))
            base.update(en)
        except Exception:
            pass
    return base

# ---------------------
# Script entrypoint
# ---------------------
def preprocess_file(input_path: str, output_path: str, remove_stopwords_flag: bool = False):
    print(f"[INFO] Preprocessing\n  input : {input_path}\n  output: {output_path}\n  remove_stopwords: {remove_stopwords_flag}")
    df = safe_read_csv(input_path)

    # Expecting columns: 'comment' and 'label' (if label missing, try to infer)
    if "comment" not in df.columns:
        # try lowercase names
        cols = [c.lower() for c in df.columns]
        if "comment" in cols:
            # rename that column to 'comment'
            df.columns = [c.lower() for c in df.columns]
        else:
            raise ValueError("Input CSV must contain a 'comment' column. Found columns: " + ", ".join(df.columns))

    # check label column
    if "label" not in df.columns:
        # if there's a second column, assume it as label
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]: "label"})
        else:
            # create a placeholder label if missing
            df["label"] = "unknown"

    stopwords_set = get_stopwords() if remove_stopwords_flag else None

    # Clean comments
    cleaned = []
    print("[INFO] Cleaning comments...")
    for raw in tqdm(df["comment"].astype(str).tolist(), total=len(df)):
        try:
            c = clean_text(raw, stopwords_set=stopwords_set, remove_stopwords_flag=remove_stopwords_flag)
        except Exception as e:
            c = ""
        cleaned.append(c)

    df["comment_clean"] = cleaned
    # drop rows where comment_clean is empty
    before = len(df)
    df = df[df["comment_clean"].str.strip() != ""].copy()
    after = len(df)
    dropped = before - after

    # Save only relevant columns (comment_clean -> comment) and label
    out_df = df[["comment_clean", "label"]].rename(columns={"comment_clean": "comment"})
    ensure_dirs_for_file(output_path)
    out_df.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"[INFO] Saved cleaned file: {output_path} (rows saved: {len(out_df)}, dropped empty: {dropped})")

    # Print small sample
    print("\n[INFO] Sample cleaned rows:")
    print(out_df.head(SAMPLE_ROWS_TO_PRINT).to_string(index=False))

# ---------------------
# CLI
# ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to raw CSV (with 'comment' column)")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output cleaned CSV path")
    p.add_argument("--remove-stopwords", action="store_true", help="Enable stopword removal (optional, uses nltk)")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        preprocess_file(args.input, args.output, remove_stopwords_flag=args.remove_stopwords)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
