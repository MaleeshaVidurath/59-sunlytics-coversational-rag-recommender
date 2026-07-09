"""
Tweet sentiment preprocessing script.
Input : ../sentiment_data_set/tweet_eval_{train|val|test}.csv
Output: tweet_eval_{train|val|test}_clean.csv  (same folder as this script)

Preprocessing steps (applied in order):
  1. Decode HTML entities
  2. Replace URLs with [URL]
  3. Replace @user mentions with [USER]
  4. Strip # from hashtags — keep the word
  5. Normalise repeated punctuation  (!!!  ->  !  ,  ???  ->  ?)
  6. Collapse multiple spaces / tabs into one space
  7. Strip leading and trailing whitespace
  8. Drop rows whose cleaned text is fewer than 3 words
"""

import re
import html
import os
import pandas as pd
from collections import Counter

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, '..', 'sentiment_data_set')
OUTPUT_DIR = SCRIPT_DIR

SPLITS = {
    'train': 'tweet_eval_train.csv',
    'val':   'tweet_eval_val.csv',
    'test':  'tweet_eval_test.csv',
}

OUTPUT_NAMES = {
    'train': 'tweet_eval_train_clean.csv',
    'val':   'tweet_eval_val_clean.csv',
    'test':  'tweet_eval_test_clean.csv',
}

# ── preprocessing function ─────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    # 1. decode HTML entities  (&amp; -> &  etc.)
    text = html.unescape(text)

    # 2. replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # 3. replace @user mentions (already anonymised in tweet_eval as "@user")
    text = re.sub(r'@\w+', '[USER]', text)

    # 4. strip # from hashtags — keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # 5. normalise repeated punctuation  (3+ -> 1)
    text = re.sub(r'([!?])\1{2,}', r'\1', text)
    # keep ellipsis (…/...) but collapse longer runs
    text = re.sub(r'\.{4,}', '...', text)

    # 6. collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # 7. strip
    text = text.strip()

    return text


def word_count(text: str) -> int:
    return len(text.split())


# ── main ───────────────────────────────────────────────────────────────────────
summary_rows = []

for split, filename in SPLITS.items():
    input_path  = os.path.join(INPUT_DIR,  filename)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAMES[split])

    df = pd.read_csv(input_path)
    rows_before = len(df)

    # apply preprocessing
    df['text'] = df['text'].fillna('').apply(preprocess)

    # drop rows with fewer than 3 words after cleaning
    too_short   = (df['text'].apply(word_count) < 3).sum()
    df = df[df['text'].apply(word_count) >= 3].reset_index(drop=True)
    rows_after  = len(df)
    dropped     = rows_before - rows_after

    # label distribution after cleaning
    dist = Counter(df['label_name'].tolist())

    df.to_csv(output_path, index=False)

    summary_rows.append({
        'split':        split,
        'rows_before':  rows_before,
        'rows_dropped': dropped,
        'rows_after':   rows_after,
        'negative':     dist.get('negative', 0),
        'neutral':      dist.get('neutral',  0),
        'positive':     dist.get('positive', 0),
    })

    print(f'[{split.upper()}]  {rows_before:,} -> {rows_after:,}  (dropped {dropped})')
    for label in ['negative', 'neutral', 'positive']:
        pct = dist.get(label, 0) / rows_after * 100
        print(f'  {label:10s}: {dist.get(label,0):,}  ({pct:.1f}%)')
    print(f'  Saved -> {OUTPUT_NAMES[split]}')
    print()

# ── per-step counts on train (for documentation) ──────────────────────────────
print('=== Step-by-step effect on train set ===')
df_raw = pd.read_csv(os.path.join(INPUT_DIR, SPLITS['train']))
df_raw['text'] = df_raw['text'].fillna('')

steps = [
    ('Raw',                      df_raw['text'].copy()),
]

t = df_raw['text'].copy()
t = t.apply(html.unescape);                                        steps.append(('After HTML decode',          t.copy()))
t = t.apply(lambda x: re.sub(r'https?://\S+|www\.\S+','[URL]',x));steps.append(('After URL replace',          t.copy()))
t = t.apply(lambda x: re.sub(r'@\w+','[USER]',x));                steps.append(('After @mention replace',     t.copy()))
t = t.apply(lambda x: re.sub(r'#(\w+)',r'\1',x));                  steps.append(('After hashtag strip',        t.copy()))
t = t.apply(lambda x: re.sub(r'([!?])\1{2,}',r'\1',x));           steps.append(('After repeated punc normalise',t.copy()))
t = t.apply(lambda x: re.sub(r'[ \t]+',' ',x).strip());           steps.append(('After whitespace normalise', t.copy()))

mask_short = t.apply(word_count) < 3
steps.append((f'After drop <3 words ({mask_short.sum()} dropped)', t[~mask_short].copy()))

for name, series in steps:
    print(f'  {name:<40} {len(series):,} rows')

print('\nDone. All clean CSVs saved to:', OUTPUT_DIR)
