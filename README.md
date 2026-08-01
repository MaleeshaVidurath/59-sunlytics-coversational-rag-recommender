# RAG-Powered Explainable Conversational Recommender System

> Final Year Research Project — Group Sunlytics (59)  
> Faculty of Information Technology, University of Moratuwa  
> Supervised by Dr. T.M. Thanthriwatta

---

## Team

| Member                 | Index   | Module             | Role                                                  |
|------------------------|---------|--------------------|-------------------------------------------------------|
| Gunarathna A.M.V.      | 214070G | M2 — Multimodal RAG| CLIP embeddings, FAISS retrieval, BLIP verification   |
| Weerathunge W.M.C.M.B. | 214225M | M3 — Adaptive RAG  | Hallucination guard, explanation memory, Streamlit UI |
| Perera M.I.V.          | 214149H | M1 — Graph RAG     | Knowledge graph, path verbalisation, NLI faithfulness |

---

## Project Overview

This system is the first RAG-powered Conversational Recommender System that accompanies every recommendation with a verified, hallucination-free natural language justification. It unifies three RAG pipelines:

- **M1 — Graph RAG**: Retrieves multi-hop KG reasoning paths and converts them into natural language explanations
- **M2 — Multimodal RAG**: Retrieves image and text evidence, verifies visual claims against product images using BLIP
- **M3 — Adaptive RAG**: Per-turn retrieval trigger, NLI hallucination guard, and explanation memory for cross-turn coherence

**Dataset**: H&M Personalized Fashion Recommendations (Kaggle)

---

## Repository Structure

sunlytics-rag-recommender/
├── README.md
├── requirements.txt
├── .gitignore
│
├── m1_graph_rag/
│   ├── kg_construction.py
│   ├── path_retrieval.py
│   ├── path_verbalisation.py
│   ├── hallucination_guard.py
│   └── notebooks/
│       └── M1_graph_rag.ipynb
│
├── m2_multimodal_rag/
│   ├── clip_embeddings.py
│   ├── faiss_index.py
│   ├── retrieval.py
│   ├── blip_verification.py
│   ├── data_preprocessing.py
│   └── notebooks/
│       └── M2_multimodal_rag.ipynb
│
├── m3_implementation/
│   ├── api/                          # FastAPI app + routers (chat, sessions, auth)
│   ├── adaptive_rag/
│   │   └── distilbert_training/      # per-turn retrieval trigger classifier
│   ├── memory/
│   │   ├── core/                     # pipeline, enrichment, CSE, preferences,
│   │   │                             #   contradiction detection
│   │   ├── db/                       # MongoDB + Redis clients
│   │   └── models/schemas.py
│   ├── text_rag/
│   │   ├── core/
│   │   │   ├── evidence_assembler.py     # action → evidence bundle
│   │   │   ├── personalized_ranker.py    # per-user selection + "why" reasons
│   │   │   ├── response_generator.py     # action-specific prompts
│   │   │   ├── hallucination_checker.py  # NLI + exact-value gates
│   │   │   └── rag_pipeline.py           # generate → check → regenerate loop
│   │   └── db/
│   │       ├── postgres_client.py        # 41,794 articles, structured filters
│   │       ├── qdrant_client.py          # semantic search
│   │       └── article_stats.py          # offline buying statistics builder
│   └── test_result/                  # evaluation suites and results
│
├── frontend/                         # React + Vite chat UI
│
├── shared/
│   ├── config.py
│   ├── data_loader.py
│   └── utils.py
│
└── app/
    └── main.py

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sunlytics59/sunlytics-rag-recommender.git
cd sunlytics-rag-recommender
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API and download dataset

```bash
# Place your kaggle.json at ~/.kaggle/kaggle.json first, then:
kaggle competitions download \
  -c h-and-m-personalized-fashion-recommendations \
  -f articles.csv -p ./data/

kaggle competitions download \
  -c h-and-m-personalized-fashion-recommendations \
  -f customers.csv -p ./data/

kaggle competitions download \
  -c h-and-m-personalized-fashion-recommendations \
  -f transactions_train.csv -p ./data/
```

### 5. Configure paths

Edit `shared/config.py` and set `DATA_DIR` to your local data folder.

### 6. Build the databases (M3)

M3 needs PostgreSQL, Qdrant, MongoDB and Redis running locally.

```bash
cd m3_implementation

# Articles into PostgreSQL + vectors into Qdrant (~10-15 min for indexing)
python -m text_rag.core.rag_pipeline --setup

# Per-customer purchase profiles into MongoDB
python -m memory.core.customer_profile_loader

# Per-article buying statistics — required by the personalised ranker
python -m text_rag.db.article_stats --build
```

The last step creates the `article_stats` and `group_stats` tables that drive popularity,
age-group matching and the "why this for you" reasons. **If it is skipped**, the system
still runs: the ranker logs a warning and falls back to user-history and semantic signals
only, losing the buying-statistics half of the ranking. Rebuild with `--force` after any
CSV change.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code only — never push directly |
| `develop`         | Integration branch — merge here before main |
| `m1/feature-name` | Member 1 (Perera) working branches |
| `m2/feature-name` | Member 2 (Gunarathna) working branches |
| `m3/feature-name` | Member 3 (Weerathunge) working branches |

### Daily workflow

```bash
# 1. Always pull latest before starting work
git checkout develop
git pull origin develop

# 2. Create your feature branch
git checkout -b m2/blip-verification

# 3. Work, then commit with clear messages
git add m2_multimodal_rag/blip_verification.py
git commit -m "feat(m2): add BLIP visual consistency check"

# 4. Push and open a Pull Request → develop
git push origin m2/blip-verification
```

### Commit message format

```
feat(m2): add CLIP image encoder
fix(m2): handle missing images in H&M dataset
docs(m2): update FAISS index setup instructions
refactor(m3): simplify adaptive trigger logic
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Text + image embeddings | CLIP (openai/clip-vit-base-patch32) |
| Visual verification | BLIP (Salesforce/blip-image-captioning-base) |
| LLM (explanations) | LLaMA-3 (Meta) |
| Hallucination detection | BART-MNLI |
| Vector database | FAISS |
| Orchestration | LangChain |
| Frontend | Streamlit |
| Deep learning | PyTorch + Transformers (HuggingFace) |

---

## Hardware Requirements

- RAM: 16 GB minimum
- GPU: NVIDIA with 8 GB+ VRAM (for CLIP, BLIP, LLaMA-3)
- Storage: 50 GB+ for dataset, models, and FAISS index

---

## Key Novel Contributions

1. **Multimodal Retrieval Framework** — unified image + text embeddings in a single FAISS vector space
2. **Visual Faithfulness Verification** — BLIP-based VLM check that generated explanations match actual product images
3. **Adaptive Retrieval Trigger** — per-turn decision whether to retrieve or reuse cached evidence
4. **NLI Hallucination Guard** — sentence-level entailment check before any response reaches the user
5. **Explanation Memory** — cross-turn coherence tracking to prevent contradictions
6. **Traceable Personalised Ranking** — once hard filters are satisfied every candidate is
   equally valid, so selection is decided by a transparent scorer blending the user's own
   purchase behaviour with per-article buying statistics (popularity, age-group affinity,
   repeat rate). Every signal that fires records a plain-language reason containing a real
   statistic, and those reasons are what the user sees, what the "why did you recommend
   this" answer cites, and what the hallucination checker verifies — so the justification
   is a trace of the actual decision rather than a story reconstructed afterwards.
   See [`PERSONALIZED_RANKER_PROCESS.md`](m3_implementation/text_rag/core/PERSONALIZED_RANKER_PROCESS.md)

---

## References

See the full reference list in the project presentation PDF.

---

## License

For academic use only — University of Moratuwa, 2025.