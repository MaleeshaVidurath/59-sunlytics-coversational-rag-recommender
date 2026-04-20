# =============================================================================
# dataset.py
# =============================================================================
# Defines the PyTorch Dataset class that wraps your CSV data.
#
# Why do we need a custom Dataset class?
# PyTorch's DataLoader (which feeds batches to the model during training)
# requires data to be wrapped in a class that implements three methods:
#   __len__  → how many samples are there?
#   __getitem__ → given an index, return one sample as a dict of tensors
#
# The Dataset class also applies tokenization here, which means tokenization
# happens once per sample, not on every epoch — that is more efficient.
# =============================================================================

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import DistilBertTokenizer
from config import PRETRAINED_MODEL, MAX_LEN, LABEL_NAMES


class RetrievalDataset(Dataset):
    """
    Wraps the CSV split files into a format PyTorch's DataLoader understands.

    Each item returned by __getitem__ is a dictionary containing:
      - input_ids      : token IDs (integers) the model reads
      - attention_mask : 1 for real tokens, 0 for padding tokens
      - labels         : the integer class label (0-7)
    """

    def __init__(self, csv_path: str, tokenizer: DistilBertTokenizer):
        """
        Args:
            csv_path  : path to one of the split CSV files (train/val/test)
            tokenizer : a pre-loaded DistilBertTokenizer instance
                        (we pass it in rather than creating it here so that
                         train/val/test all share the exact same tokenizer)
        """
        self.tokenizer = tokenizer

        # Load the CSV — we only need two columns: input_text and label
        df = pd.read_csv(csv_path)
        self.texts  = df["input_text"].tolist()
        self.labels = df["label"].tolist()

        # Sanity check: labels must be integers in range [0, NUM_LABELS-1]
        assert all(0 <= l < len(LABEL_NAMES) for l in self.labels), \
            "Found label value outside expected range 0-7. Check your CSV."

    def __len__(self) -> int:
        """Returns the total number of samples in this split."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenizes one sample and returns it as a dict of tensors.

        The tokenizer does three things here:
          1. Splits the text into subword tokens (WordPiece tokenization)
          2. Converts each token to its integer ID from the vocabulary
          3. Pads or truncates to MAX_LEN so all sequences are the same length

        truncation=True  → sequences longer than MAX_LEN are cut off
        padding="max_length" → sequences shorter than MAX_LEN are padded with 0s
        return_tensors="pt" → return PyTorch tensors, not Python lists
        """
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            # squeeze(0) removes the extra batch dimension the tokenizer adds
            # e.g. shape [1, 256] becomes [256]
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
