# RAG-Powered Explainable Conversational Recommender System

> Final Year Research Project — Group Sunlytics (59)  
> Faculty of Information Technology, University of Moratuwa  
> Supervised by Dr. T.M. Thanthriwatta

---

## Team

| Member                 | Index   | Module             | Role                                                  |
|------------------------|---------|--------------------|-------------------------------------------------------|
| Gunarathna A.M.V.      | 214070G | M2 — Multimodal RAG| Fine-tuned CLIP + FAISS retrieval, 3-layer visual verification |
| Weerathunge W.M.C.M.B. | 214225M | M3 — Adaptive RAG  | Hallucination guard, explanation memory, React UI     |
| Perera M.I.V.          | 214149H | M1 — Graph RAG     | Knowledge graph, path verbalisation, NLI faithfulness |

---

## Project Overview

This system is the first RAG-powered Conversational Recommender System that accompanies every recommendation with a verified, hallucination-free natural language justification. It unifies three RAG pipelines:

- **M1 — Graph RAG**: Retrieves multi-hop KG reasoning paths and converts them into natural language explanations
- **M2 — Multimodal RAG**: Retrieves image and text evidence with a domain fine-tuned CLIP space, then verifies visual claims against the actual product images through a 3-layer guard
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
│   ├── backend/                        # FastAPI service (port 8001)
│   │   ├── main.py                     #   /api/process, /api/images/{article_id}
│   │   └── schemas.py                  #   typed M3 → M2 request contract
│   ├── m2_action_router.py             # dispatcher for the 5 M2 actions
│   ├── m2_handlers.py                  # per-action retrieval + generation logic
│   ├── clip_embeddings.py              # fine-tuned CLIP text/image encoder
│   ├── faiss_index.py                  # FAISS search over the shared vector space
│   ├── cross_encoder_reranker.py       # ms-marco MiniLM reranking of candidates
│   ├── diversity_bandit.py             # Thompson-sampling diversity control
│   ├── llm_generator.py                # Groq-hosted Llama text + vision calls
│   ├── vlm_kansei.py                   # VLM Kansei impression of product images
│   ├── hallucination_guard/            # 3-layer explanation verification
│   │   ├── layer_1_knowledge_self_reflection.py
│   │   ├── layer_2_cove_verification.py
│   │   ├── layer_3_vlm_visual_verification.py
│   │   ├── clip_faithfulness_scorer.py
│   │   └── regeneration_loop.py
│   ├── knowledge_base/                 # curated Kansei fashion knowledge base
│   │   ├── fashion_kb.py
│   │   └── kb_retriever.py
│   ├── collaborative_filtering/        # NCF cold-start scorer + trained factors
│   │   ├── cf_scorer.py
│   │   └── models/                     #   item_factors.npy, popularity.npy, ...
│   ├── vector_db/                      # FAISS index + article_id mapping
│   ├── finetune_clip/                  # Kaggle CLIP fine-tuning + index rebuild
│   └── evaluation/                     # novelty evaluations, figures, results
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

### 7. Run M2 standalone (no databases required)

M2 is a self-contained FastAPI service. It needs **no PostgreSQL, Qdrant, MongoDB or Redis** —
those are M3's — so it can be started and exercised on its own:

```bash
# from the repository root
uvicorn m2_multimodal_rag.backend.main:app --port 8001
```

Then open **http://localhost:8001/docs** and call `POST /api/process` directly.

Requirements: a `GROQ_API_KEY` in `.env`, the FAISS artifacts
(`m2_multimodal_rag/vector_db/m2_clip_faiss.bin` + `m2_faiss_mapping.csv`), the trained CF
factors in `collaborative_filtering/models/`, and the H&M images under `data/`.

`POST /api/process` takes the typed M3 → M2 contract defined in
[`backend/schemas.py`](m2_multimodal_rag/backend/schemas.py) — a `retrieval_input` whose `payload`
is resolved by its `action` field, one of `catalog_search`, `item_attribute_lookup`,
`item_compare`, `explanation_generate`, `item_detail_lookup`. `GET /api/images/{article_id}`
serves the product image for a result.

### 8. Run the full stack

Three processes: M2 (8001), M3 (8000), frontend (Vite).

```bash
uvicorn m2_multimodal_rag.backend.main:app --port 8001    # terminal 1
cd m3_implementation && uvicorn api.main:app --port 8000  # terminal 2
cd frontend && npm install && npm run dev                 # terminal 3
```

The frontend talks to M3 at `http://localhost:8000`; M3 reaches M2 via `M2_MULTIMODAL_URL`.
This has **no default** — set it in `.env` or M3 will silently skip the M2 call:

```
M2_MULTIMODAL_URL=http://127.0.0.1:8001
```

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
git checkout -b m2/vlm-verification

# 3. Work, then commit with clear messages
git add m2_multimodal_rag/hallucination_guard/layer_3_vlm_visual_verification.py
git commit -m "feat(m2): add VLM visual consistency check"

# 4. Push and open a Pull Request → develop
git push origin m2/vlm-verification
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
| Text + image embeddings (M2) | `open_clip` ViT-B-32, `laion2b_s34b_b79k` weights, domain fine-tuned on H&M |
| Candidate reranking (M2) | Cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Cold-start scoring (M2) | Neural collaborative filtering (trained item factors) |
| Visual verification (M2) | ViLT VQA (`ViltForQuestionAnswering`) + CLIP image–text faithfulness scorer |
| LLM — explanations (M2) | Groq-hosted `llama-3.1-8b-instant` (text), `llama-4-scout-17b-16e-instruct` (vision) |
| Hallucination detection | M2: 3-layer guard (self-reflection → CoVe → VLM) · M3: NLI entailment gates |
| Vector database | FAISS (M2) · Qdrant (M3) |
| Frontend | React + Vite |
| Deep learning | PyTorch + Transformers (HuggingFace) |

---

## Hardware Requirements

- RAM: 16 GB minimum
- GPU: NVIDIA with 8 GB+ VRAM (for CLIP, ViLT and the local NLI models; M2's Llama calls are
  served remotely by Groq and need only a `GROQ_API_KEY`)
- Storage: 50 GB+ for dataset, models, and FAISS index

---

## Key Novel Contributions

### M2 — Multimodal RAG

1. **Domain fine-tuned multimodal retrieval** — image + text embeddings in a single FAISS space,
   with CLIP fine-tuned on H&M rather than used off the shelf.
   *Recall@10 57.2% → 76.0% (+18.7 pp, 95% CI [+16.8, +20.6], n=2084).*
2. **Visual faithfulness verification** — a 3-layer guard (knowledge-grounded self-reflection →
   Chain-of-Verification → ViLT VQA visual check) with a CLIP image–text faithfulness scorer and a
   regeneration loop, so no claim reaches the user without being checked against the product image.
   *Visual-corruption recall 0.76 → 0.80 over a text-only first layer (n=168).*
3. **NCF cold-start recovery** — collaborative-filtering signal fused into ranking for users with
   no usable purchase history. *Cold-start Hit@10 4.3% → 19.6% (n=46).*
4. **Thompson-sampling diversity control** — the relevance/diversity trade-off adapts to in-session
   rejections instead of using a fixed weight. *λ 0.70 → 0.51 across 10 rejections (n=200).*
5. **Kansei fashion knowledge base** — grounds affective/stylistic language in curated fashion
   knowledge rather than letting the LLM invent it.
   *77% win rate in a blind LLM-judge comparison, 23W/7L/0T, 95% CI [59%, 88%] (n=30).*

Evaluation detail: [`m2_multimodal_rag/evaluation/results/SUMMARY.md`](m2_multimodal_rag/evaluation/results/SUMMARY.md)
· figures in [`evaluation/results/figures/`](m2_multimodal_rag/evaluation/results/figures/)

### M3 — Adaptive RAG

6. **Adaptive Retrieval Trigger** — per-turn decision whether to retrieve or reuse cached evidence
7. **NLI Hallucination Guard** — sentence-level entailment check before any response reaches the user
8. **Explanation Memory** — cross-turn coherence tracking to prevent contradictions
9. **Traceable Personalised Ranking** — once hard filters are satisfied every candidate is
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