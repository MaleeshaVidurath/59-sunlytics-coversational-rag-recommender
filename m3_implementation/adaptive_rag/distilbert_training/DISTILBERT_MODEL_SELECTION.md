# DistilBERT — Model Selection & Architecture
### Sunlytics CRS — M3 Adaptive RAG Module

This document explains what DistilBERT is, why it was selected for the 8-class
intent classifier, and why all alternative approaches were rejected.

---

## 1. What is BERT

BERT (Bidirectional Encoder Representations from Transformers) is a language model
published by Google in 2018. It was pre-trained on massive text corpora (Wikipedia +
BookCorpus) to understand the meaning of words **in context** — reading every sentence
in both directions simultaneously using self-attention.

Before BERT, most NLP models read text left-to-right or right-to-left. BERT reads
the entire sentence at once, so the meaning of each word is informed by every other
word around it. For example, the word *"bank"* in *"river bank"* and *"bank account"*
gets different contextual embeddings because BERT sees the full surrounding context.

After pre-training, BERT can be fine-tuned on any NLP task (classification, question
answering, named entity recognition) with a small amount of task-specific labelled
data. This makes it extremely practical for research — the expensive language
understanding is already learned; only the task-specific layer needs training.

---

## 2. What is DistilBERT

DistilBERT is a **smaller, faster version of BERT** created by Hugging Face in 2019
using a technique called **knowledge distillation**.

### Knowledge Distillation

A large "teacher" model (BERT) trains a smaller "student" model (DistilBERT) to
mimic its behaviour. The student does not just copy the final labels — it learns to
reproduce the internal probability distributions of the teacher across all outputs.
This transfers the teacher's language understanding into a model with half the layers.

### BERT vs DistilBERT

| Property | BERT | DistilBERT |
|----------|------|------------|
| Parameters | 110M | 66M |
| Transformer layers | 12 | 6 |
| Inference speed | baseline | 60% faster |
| Model size | baseline | 40% smaller |
| Performance retained | 100% | 97% |

DistilBERT retains 97% of BERT's language understanding at 40% of the size and
60% faster inference — making it practical for real-time use in a production system
where classification runs on every conversation turn.

---

## 3. How DistilBERT Works for Classification

The 8-class intent classifier adds a linear classification head on top of DistilBERT.
The full forward pass is:

```
Input text  e.g. "USER: show me jackets [SEP] BOT: here are two... [SEP] CURRENT: cheaper ones?"
     ↓
DistilBERT tokeniser → token IDs + attention mask
     ↓
6 transformer layers (self-attention + feed-forward)
→ contextual embedding vector per token
     ↓
[CLS] token embedding  (384-dim vector — summary of entire input)
     ↓
Linear classification head  (384 → 8 logits)
     ↓
Softmax → probability distribution over 8 intent classes
     ↓
argmax → predicted label  e.g. REFINEMENT (label 1)
```

The `[CLS]` (classification) token is prepended to every input during tokenisation.
After all 6 transformer layers, the `[CLS]` embedding represents the meaning of the
entire sequence. The classification head reads only this one vector to make its
prediction — all 8 class scores come from this single 384-dim summary vector.

---

## 4. Why DistilBERT Was Selected

### The Core Reason — [SEP] Token Alignment with Pre-training

The input format used in this system is:

```
USER: <prev_msg> [SEP] BOT: <bot_response> [SEP] CURRENT: <current_message>
```

`[SEP]` is not just a delimiter — it is a **native special token in DistilBERT's
pre-training vocabulary**. BERT and DistilBERT were pre-trained on sentence-pair
tasks using `[SEP]` as the segment boundary token between two sentences. By
structuring the conversation context with `[SEP]`, the model's pre-trained
understanding of segment transitions is directly leveraged — the model already
encodes the boundary between speakers as a meaningful structural signal.

This alignment between the input format and the model's pre-training is a
deliberate architectural decision. RoBERTa, for example, does not use
`token_type_ids` and handles `[SEP]` differently — this alignment would be lost.

### Additional Reasons

**Fast local inference — no API dependency**
DistilBERT classifies in under 50ms locally with no external API call. In a
production CRS where classification runs on every single user turn, latency
matters. The classifier must not become a bottleneck in the pipeline.

**Proven on intent classification**
Fine-tuned BERT-family models are the established standard for intent classification
in dialogue systems. Multiple published benchmarks (ATIS, SNIPS, MultiWOZ) show
BERT-family models outperforming all alternatives on short-text intent tasks.

**Fine-tuning paradigm fits the training setup**
DistilBERT fine-tuning with a `SequenceClassification` head requires only adding
a linear layer on top of the pre-trained model and training on labelled examples.
The 52,028 synthetic training samples (later mixed with 24,000 real SIMMC samples)
are sufficient for reliable fine-tuning — far more than needed.

---

## 5. Why Other Approaches Were Rejected

### Traditional ML — SVM, Logistic Regression, TF-IDF

These require hand-crafted features and treat text as a bag of words. The intent
classes in this system have subtle boundaries — REFINEMENT and FEEDBACK can be
expressed with nearly identical surface forms (*"too expensive"* is FEEDBACK;
*"show me something cheaper"* is REFINEMENT). TF-IDF cannot capture this
contextual distinction. The 6 user personas introduce extreme linguistic variation
(*"ngl first one"* vs *"Could you please show me the first option?"*) — keyword
features collapse across personas entirely.

### Rule-based / Keyword Matching

Fragile and unscalable across 6 writing styles. Would require maintaining separate
rule sets per persona. Cannot generalise to unseen phrasings. A user saying
*"lowkey need smth cheaper"* (teenager) and *"I was thinking something more
affordable"* (polite shopper) express the same REFINEMENT intent — no single
keyword rule captures both.

### LSTM / BiLSTM

Pre-transformer sequential models with no pre-trained language understanding to
leverage. Must learn language representations from scratch on the training data
alone, requiring far more data and longer training. Also struggle with long-range
dependencies — the relationship between the prior bot response and the current user
message spans many tokens, which BiLSTMs handle weakly compared to transformer
self-attention.

### Full BERT (bert-base-uncased)

DistilBERT retains 97% of BERT's performance at 40% of the size and 60% faster
inference. Full BERT adds latency without meaningful accuracy gain on an 8-class
short-text classification task where the performance difference between BERT and
DistilBERT is negligible.

### RoBERTa

RoBERTa is a stronger model than DistilBERT but is 3× larger with significantly
higher inference cost. More critically, RoBERTa was pre-trained **without**
`token_type_ids` — it does not use segment embeddings. The `[SEP]` token in the
input format would be treated as an ordinary token rather than a segment boundary,
losing the structural signal that DistilBERT encodes natively. For short
conversational text classification, the accuracy gap over DistilBERT does not
justify the inference overhead or the architectural mismatch.

### Large Language Models (GPT, Claude, Llama — zero-shot)

LLMs can classify text zero-shot but are unreliable for 8 classes with precisely
defined boundaries — particularly the REFINEMENT/FEEDBACK boundary which requires
understanding that REFINEMENT implies a new request while FEEDBACK is a pure
reaction with no new request. Zero-shot LLM classification on ambiguous cases
produces inconsistent results without fine-tuning.

Furthermore, LLM inference adds 500ms–2s per classification call compared to under
50ms for DistilBERT locally. The system already uses Groq for response generation
— adding LLM dependency for classification would create a cascading failure point:
if the Groq API is unavailable, both classification and generation would fail.
DistilBERT runs entirely locally with no API dependency.

### Sentence-BERT (SBERT)

SBERT is optimised for semantic similarity tasks (sentence pairs), not sequence
classification. It would still require a custom classification head and fine-tuning
on the labelled training data — the same training pipeline as DistilBERT. Since
both require fine-tuning, DistilBERT's direct `[CLS]` token classification is the
cleaner and more established approach.

### T5-large

T5 (Text-to-Text Transfer Transformer) was not selected for five specific reasons:

**1. Wrong architecture for classification — encoder-decoder vs encoder-only**
T5 uses a full encoder-decoder architecture designed for generative tasks —
translation, summarisation, question answering. For classification, only the encoder
is needed to produce a representation of the input. The decoder is entirely wasted
computation. DistilBERT is encoder-only — it reads the input once, produces a
`[CLS]` embedding, and passes it to a classification head. T5's decoder adds
inference overhead with zero benefit for a fixed-label classification task.

**2. Classification by text generation — not by logits**
T5 treats classification as a text generation task. To classify an input it would
generate a text string like `"REFINEMENT"` or `"INITIAL_REQUEST"` token by token
using autoregressive decoding — one token at a time until the full label name is
produced. DistilBERT produces 8 logits in a single forward pass — one number per
class, returned instantly. T5's generative approach is slower and less reliable for
a fixed label set because nothing prevents it from generating an invalid or
misspelled label name.

**3. T5-large is 11× larger than DistilBERT**

| Model | Parameters | VRAM to load |
|-------|-----------|-------------|
| DistilBERT | 66M | ~250MB |
| T5-small | 60M | ~240MB |
| T5-base | 220M | ~900MB |
| T5-large | 770M | ~3GB |

T5-large at 770M parameters requires ~3GB GPU VRAM just to load. In a local
research prototype where the classifier runs on every conversation turn alongside
Qdrant, PostgreSQL, MiniLM, and DeBERTa already loaded in memory, T5-large is
not deployable on a standard development machine.

**4. Pre-training does not align with the [SEP] input format**
T5 uses SentencePiece tokenisation and was pre-trained on a span-corruption
denoising task — randomly masking spans of text and learning to reconstruct them.
It has no concept of `[SEP]` as a segment boundary token. BERT and DistilBERT were
pre-trained specifically with `[SEP]` separating sentence pairs — using `[SEP]` to
separate conversation turns (`USER: ... [SEP] BOT: ... [SEP] CURRENT: ...`) aligns
naturally with DistilBERT's pre-trained segment understanding. This alignment is
completely absent in T5.

**5. Designed for the wrong task family**
T5 was designed for generative tasks. Intent classification is a discriminative
task — the model must assign a label from a fixed closed set, not generate free
text. Using a generative architecture for a discriminative task is an architectural
mismatch. T5-large would be a valid choice for tasks like generating explanations
or summarising session history — which is exactly why the system uses a generative
LLM (Groq/Llama) for response generation. For classification, DistilBERT is the
correct architecture.

**T5 vs DistilBERT comparison:**

| Criterion | DistilBERT | T5-large |
|-----------|-----------|---------|
| Architecture | Encoder-only — correct for classification | Encoder-decoder — decoder wasted |
| Classification method | 8 logits in one forward pass | Generates label text token by token |
| Parameters | 66M | 770M (11× larger) |
| VRAM | ~250MB | ~3GB |
| [SEP] alignment | Native pre-training token | Not applicable |
| Designed for | Understanding + classification | Generative tasks |
| Inference speed | <50ms | Seconds (autoregressive decoding) |

---

## 6. Summary Comparison

| Approach | Why Rejected |
|----------|-------------|
| SVM / Logistic Regression | No contextual understanding, collapses across 6 personas |
| Rule-based | Cannot generalise across personas and unseen phrasings |
| LSTM / BiLSTM | No pre-trained language model, weak long-range context |
| Full BERT | 60% slower inference, negligible accuracy gain on this task |
| RoBERTa | No token_type_ids — [SEP] alignment lost, 3× larger |
| LLM zero-shot | Unreliable on ambiguous boundaries, high latency, API dependency |
| SBERT | Optimised for similarity not classification, same training cost |
| T5-large | Encoder-decoder wasted on classification, generative output unreliable, 11× larger, no [SEP] alignment, ~3GB VRAM |
| **DistilBERT** | **[SEP] pre-training alignment, fast local inference, proven on intent classification, 97% of BERT at 40% size** |

---

## 7. Model Files

| File | Location |
|------|----------|
| Training script | `distilbert_training/train.py` |
| Inference script | `distilbert_training/predict.py` |
| Trained model | `adaptive_rag/models/intent_classifier/` |
| Training data (synthetic) | `distilbert_training/data/v4_train_midSession.csv` |
| Training data (mixed real+synthetic) | `distilbert_training/data/v5_train_mixed.csv` |
| Preprocessing doc | `distilbert_training/SIMMC_PREPROCESSING.md` |
| Synthetic data generation doc | `distilbert_training/SYNTHETIC_DATA_GENERATION.md` |
| Ensemble approach doc | `distilbert_training/REAL_DATA_ENSEMBLE_APPROACH.md` |
