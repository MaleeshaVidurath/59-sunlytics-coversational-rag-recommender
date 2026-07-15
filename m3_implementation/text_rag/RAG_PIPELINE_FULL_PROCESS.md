# Text RAG Pipeline — Full Process Documentation
### Sunlytics CRS — M3 Implementation

This document covers the complete end-to-end process of the Text RAG pipeline:
the retrieval architecture, Qdrant vector database, PostgreSQL structured database,
evidence assembly, response generation, hallucination checking, and contradiction detection.

---

## 1. Architecture Overview

The Text RAG pipeline takes a classified user intent and converts it into a verified,
grounded natural language response. It uses two databases working together:

| Component | Role |
|-----------|------|
| **Qdrant** | Semantic vector search — finds which articles are relevant to the user query |
| **PostgreSQL** | Structured retrieval — fetches full article details for those articles |
| **EvidenceAssembler** | Orchestrates both databases → builds evidence bundle |
| **ResponseGenerator** | Builds action-specific prompt → Groq/Ollama LLM → response text |
| **HallucinationChecker** | MiniLM + DeBERTa NLI — verifies response against evidence |
| **ContradictionDetector** | Session graph cross-turn consistency check |

---

## 2. RAG Pipeline — Full Flow (every turn)

```
User message
     ↓
Memory Pipeline → pipeline_output
     ↓
TextRAGPipeline.process()
     ↓
Step 0  → Not relevant / session blocked?    → return refusal immediately
Step 1  → Cached recommendation available?  → serve from cache, skip databases
Step 2  → EvidenceAssembler                 → query Qdrant + PostgreSQL → evidence bundle
Step 3  → ResponseGenerator                 → build prompt → Groq LLM → response text
Step 4  → HallucinationChecker              → MiniLM + DeBERTa NLI → pass or fail
Step 5  → If fail → retry with stricter prompt (max 3 attempts)
Step 6  → ContradictionDetector             → session graph cross-turn check → correct response
Step 7  → store_response()                  → save to memory pipeline
     ↓
Return final response to user
```

### What pipeline_output contains (from Memory Pipeline)

| Field | Description |
|-------|-------------|
| `label` | Intent class — INITIAL_REQUEST, REFINEMENT, etc. |
| `retrieval_strategy` | FULL, PARTIAL, or NO |
| `retrieval_input` | Action + filters for the assembler |
| `memory_context` | User preferences, history, rejected items |
| `session_id`, `user_id`, `turn_id` | Session identifiers |

---

## 3. Per Label — What Happens in the Pipeline

### INITIAL_REQUEST — strategy: FULL
- **Action:** `catalog_search`
- Qdrant semantic search (filters applied before cosine scoring) → up to 20 candidates, preference-ranked
- PostgreSQL filtered search (same hard filters, no penalty SQL) → up to 20 candidates, preference-ranked with penalty demotion
- Both result sets merged (Qdrant first), deduplicated, top N selected
- ResponseGenerator produces a fresh recommendation list
- Hallucination checked (name, colour, price verified)
- Contradiction detector adds new product nodes to session graph
- Items stored in memory as current recommendations

### REFINEMENT — strategy: FULL
- **Action:** `catalog_search`
- Same dual Qdrant + PostgreSQL search but filters updated (new colour, price range, style)
- Previously rejected items excluded via `exclude_ids` in both searches
- Full new search — completely replaces previous results

### ATTRIBUTE_QUESTION — strategy: PARTIAL
- **Action:** `item_attribute_lookup`
- PostgreSQL fetches specific attribute (material, size, care) for one item already in session
- No Qdrant search — item already known from session context
- ResponseGenerator answers the specific attribute question only
- Hallucination check runs on the attribute answer

### EXPLANATION_WHY — strategy: PARTIAL
- **Action:** `explanation_generate`
- PostgreSQL fetches the item already shown + confirmed preference matches from memory
- ResponseGenerator explains why that item was recommended based on user's stated preferences
- No new search — uses session memory to justify the previous recommendation

### COMPARISON — strategy: PARTIAL
- **Action:** `item_compare`
- PostgreSQL fetches two items (item_a, item_b) already shown in session
- ResponseGenerator generates a structured side-by-side comparison
- Hallucination check verifies both items' name, colour, price

### SELECTION_REFERENCE — strategy: PARTIAL
- **Action:** `item_detail_lookup`
- PostgreSQL fetches full detail of one selected item (all attributes, material description)
- ResponseGenerator gives a detailed breakdown of that specific item
- Hallucination check runs on all detail fields

### FEEDBACK — strategy: NO
- **Action:** `no_retrieval`
- No Qdrant, no PostgreSQL
- Sentiment classifier (Twitter-RoBERTa) scores feedback as positive/negative/neutral
- Negative → rejected items stored in memory, excluded from future searches
- Positive → preference confirmed, item marked as accepted
- ResponseGenerator gives a conversational acknowledgement only

### CHITCHAT — strategy: NO
- **Action:** `no_retrieval`
- No Qdrant, no PostgreSQL
- ResponseGenerator gives a simple conversational reply
- Hallucination check skipped entirely (`_SKIP_HALLUCINATION_CHECK = {"no_retrieval"}`)
- Nothing stored to product memory

---

## 4. Step 0 Special Cases

| Condition | What happens |
|-----------|-------------|
| `not_relevant=True` | Returns refusal message immediately — no databases, no LLM |
| `session_context_blocked=True` | Same — blocked by memory pipeline |
| `is_cached_recommendation=True` | Serves previous recommendation from session memory cache — skips Qdrant and PostgreSQL entirely |
| `catalog_search` with 0 items | Skips LLM entirely — returns honest "no products found" message to prevent hallucination |

---

## 5. Hallucination Retry Logic

```
Attempt 1  strictness=0  → friendly, descriptive response
    ↓ fails hallucination check?
Attempt 2  strictness=1  → "copy values exactly, be accurate" + contradicted fields passed in
    ↓ fails hallucination check?
Attempt 3  strictness=2  → bullet-only, minimal description
    ↓ always accepted regardless
```

`contradicted_fields` from the failed check are injected into the next generation
prompt so the LLM knows exactly which fields to copy verbatim from evidence.

---

## 6. Qdrant Vector Database

### Purpose

Qdrant answers the question: *"Which of the 41,794 articles are semantically most
relevant to this user's natural language query?"*

It does this by comparing the meaning of the query against the meaning of every
article description — not just keyword matching.

### Fields Encoded into Vectors

`_make_article_text()` combines 8 article fields into one text string:

```python
prod_name                 +   # "Spring Wrap dress"
product_type_name         +   # "Dress"
colour_group_name         +   # "Black"
garment_group_name        +   # "Dressed"
graphical_appearance_name +   # "Solid"
index_group_name          +   # "Ladieswear"
section_name              +   # "Special Collections"
detail_desc                   # "Wrap dress in woven fabric with a V-neck..."
```

All 8 joined into one string → MiniLM (`all-MiniLM-L6-v2`) encodes → **384-dimensional vector**.

`detail_desc` is the most important field — it is free text describing the garment
and provides the richest semantic signal for matching natural language queries.

### What Is Stored Per Qdrant Point

Each of the 41,794 articles is stored as one Qdrant point with two parts:

**Vector** — the 384-dim float array (used for cosine similarity scoring)

**Payload** — full article fields stored alongside the vector (used for hard
filtering and for returning results without an extra PostgreSQL query):

```
article_id, prod_name, product_type_name, product_group_name,
colour_group_name, graphical_appearance_name,
perceived_colour_master_name, index_group_name,
garment_group_name, section_name, department_name,
detail_desc, avg_price
```

### Indexing Pipeline (done once at startup)

```
sample_articles.csv (41,794 rows)
     ↓
Read in batches of 256
     ↓
_make_article_text() → text string per article
     ↓
MiniLM.encode(batch) → 256 × 384 float matrix
     ↓
Build PointStruct(id=article_id, vector=..., payload={all fields})
     ↓
qdrant.upsert(points) → stored on disk
     ↓
Repeat until all 41,794 articles indexed
```

If `get_collection_count() > 0` on startup, indexing is skipped — vectors
persist on disk between restarts.

### Semantic Search Flow (every INITIAL_REQUEST / REFINEMENT)

```
User query string  e.g. "I want a cheap blue casual jacket"
     ↓
MiniLM.encode([query]) → 384-dim query vector
     ↓
Build Qdrant filter from payload fields:
  MUST:     colour_group_name = "Blue"        (exact match)
  MUST:     product_type_name = "Jacket"      (exact match)
  MUST:     avg_price <= 50.00                (range)
  MUST NOT: article_id IN [rejected_ids]      (exclude)
     ↓
Qdrant: cosine_similarity(query_vector, all_article_vectors)
        applied only to articles passing the filter
     ↓
Top 10 most similar articles returned, ranked by cosine score
     ↓
article_ids passed to PostgreSQL for full data fetch
```

**Why pre-filtering matters:** Qdrant filters payload fields BEFORE scoring.
Only articles matching the hard constraints are ranked. This means the cosine
similarity score determines the best match within the user's stated constraints —
not across all 41,794 articles indiscriminately.

### Physical Storage Location

Qdrant runs as a local server at `localhost:6333`. Data is stored on disk by Qdrant itself:

| Deployment method | Storage location |
|-------------------|-----------------|
| Docker | Docker volume `qdrant_storage` managed by Docker |
| Direct binary | `./storage/` folder next to `qdrant.exe` |

The project connects to it via the Python `qdrant-client` library. Qdrant is a
**separate running service**, not a file inside the project directory.

### Why Qdrant Was Selected

Six specific reasons Qdrant was chosen over alternatives:

**1. Pre-filtering before similarity scoring — the most critical reason**

Qdrant applies hard filters on payload fields **before** computing cosine similarity.
When a user says *"show me black dresses under £40"*, Qdrant first eliminates all
non-black, non-dress, over-£40 articles from the 41,794 items, then runs similarity
scoring **only on the remaining candidates**.

Other libraries like FAISS score all 41,794 vectors first and filter afterwards —
meaning a *"red shirt"* could score higher than a *"black dress"* simply because its
text embedding is closer to the query. This is semantically wrong for a CRS where
hard constraints must be respected absolutely.

**2. Free, open source, local deployment**

Qdrant runs entirely on localhost with no API key, no usage limits, and no cost
per query. Pinecone (the most popular alternative) is a cloud-only service with
a paid tier beyond free limits. For a research prototype making hundreds of queries
during evaluation, cloud cost was a real constraint. Qdrant has zero per-query cost.

**3. Persistent disk storage — no re-indexing per session**

Qdrant persists all 41,794 vectors to disk automatically. The system checks
`get_collection_count()` at startup — if vectors already exist it skips indexing
entirely. FAISS is in-memory by default, requiring the index to be rebuilt or
saved/loaded manually on every restart. Qdrant's persistence is built-in with
no extra code.

**4. Purpose-built vector database vs. extension on top of PostgreSQL**

`pgvector` (the PostgreSQL vector extension) was considered since PostgreSQL is
already used in this system. However, three problems made it unsuitable:

- **Competing for the same connection pool** — PostgreSQL uses a connection pool,
  a limited number of open database connections shared across all queries. Semantic
  search (vector similarity over 41,794 items) is a heavy, slow operation. Structured
  queries (fetch article by ID) are fast and lightweight. If both run in the same
  PostgreSQL instance, a slow vector search blocks or delays the fast structured
  queries because they are all waiting for the same connections. Qdrant runs as a
  completely separate service — its heavy vector work never touches the PostgreSQL
  connection pool at all.

- **Using the wrong tool for the job** — PostgreSQL is a relational database,
  optimised for rows, indexes, WHERE clauses, and exact lookups. Qdrant is a vector
  database, optimised for floating-point distance calculations across high-dimensional
  arrays. Using PostgreSQL for both tasks is like using a hammer for both hammering
  nails and measuring distances — it can do both, but a dedicated measuring tape is
  far better for one of them. Qdrant separates concerns cleanly: PostgreSQL handles
  structured retrieval, Qdrant handles semantic search.

- **Weaker nearest neighbour performance** — To find the most similar vectors across
  41,794 items quickly, both systems use an index structure. Qdrant uses **HNSW**
  (Hierarchical Navigable Small World) — a graph-based index specifically designed
  for fast approximate nearest neighbour search. pgvector also supports HNSW but it
  is a bolt-on addition to a general-purpose database engine. Qdrant's entire
  architecture — memory layout, threading, storage format — is built around HNSW
  from the ground up, making it consistently faster and more memory-efficient at
  the same scale.

**5. Python client with typed models**

Qdrant provides a first-class Python client (`qdrant-client`) with typed models
(`PointStruct`, `Filter`, `FieldCondition`, `Range`, `HasId`). This makes filter
construction explicit and safe — as seen in `semantic_search()`. ChromaDB and
Weaviate have Python clients but their filtering APIs are less expressive for the
combination of exact-match + range filters needed here (colour = exact match,
price = range, excluded IDs = NOT IN).

**6. Batch upsert for efficient indexing**

Qdrant's `upsert()` accepts batches of points efficiently. The indexing script
processes all 41,794 articles in batches of 256 — encoding 256 articles with
MiniLM then uploading in one API call. This is why indexing completes in 10–15
minutes rather than hours. FAISS requires building the entire index in memory
before it can be queried at all.

**Summary comparison:**

| Criterion | Qdrant | FAISS | Pinecone | pgvector |
|-----------|--------|-------|----------|----------|
| Pre-filter before scoring | Yes | No | Yes | Partial |
| Free / local | Yes | Yes | No (paid) | Yes |
| Persistent storage | Yes | Manual | Yes (cloud) | Yes |
| Python typed client | Yes | Minimal | Yes | Via psycopg2 |
| Separate from PostgreSQL | Yes | Yes | Yes | No |
| Production server | Yes | No | Yes | No |

Qdrant is the only option that satisfies all six criteria simultaneously for a
local research prototype.

---

## 7. PostgreSQL Structured Database

### Purpose

PostgreSQL answers: *"Give me the complete structured data for these specific
article IDs."* It stores all 41,794 H&M articles as structured rows with full
field detail, enabling fast lookup by ID and filtered catalog search.

### Schema

One table: `articles` with 24 columns and 7 indexes on the most-filtered fields.

```sql
CREATE TABLE articles (
    article_id                  BIGINT PRIMARY KEY,
    product_code                INTEGER,
    prod_name                   VARCHAR(50),
    product_type_no             SMALLINT,
    product_type_name           VARCHAR(30),
    product_group_name          VARCHAR(25),
    graphical_appearance_no     INTEGER,
    graphical_appearance_name   VARCHAR(25),
    colour_group_code           SMALLINT,
    colour_group_name           VARCHAR(20),
    perceived_colour_value_name VARCHAR(15),
    perceived_colour_master_name VARCHAR(20),
    department_no               INTEGER,
    department_name             VARCHAR(45),
    index_code                  CHAR(1),
    index_name                  VARCHAR(35),
    index_group_no              SMALLINT,
    index_group_name            VARCHAR(15),
    section_no                  SMALLINT,
    section_name                VARCHAR(35),
    garment_group_no            SMALLINT,
    garment_group_name          VARCHAR(35),
    detail_desc                 TEXT,
    avg_price                   NUMERIC(8,2)
);

-- Indexes for fast filtering
CREATE INDEX idx_colour    ON articles(colour_group_name);
CREATE INDEX idx_type      ON articles(product_type_name);
CREATE INDEX idx_idx_grp   ON articles(index_group_name);
CREATE INDEX idx_garment   ON articles(garment_group_name);
CREATE INDEX idx_graphical ON articles(graphical_appearance_name);
CREATE INDEX idx_price     ON articles(avg_price);
CREATE INDEX idx_section   ON articles(section_name);
```

`avg_price` is computed from `sample_transactions.csv` (average of all recorded
transaction prices per article) and stored alongside the article data.

### Four Query Functions

**`get_article_by_id(article_id)`**
Used by: `item_detail_lookup`, `item_attribute_lookup`, `explanation_generate`
```sql
SELECT * FROM articles WHERE article_id = $1
```

**`get_articles_by_ids(article_ids)`**
Used by: `item_compare`, and after Qdrant search to fetch full data
```sql
SELECT * FROM articles WHERE article_id = ANY($1::bigint[])
```
Preserves the order of the input ID list (Qdrant rank order).

**`search_articles_filtered(filters, exclude_ids, limit)`**
Used by: `catalog_search` as a parallel structured search alongside Qdrant.
Applies hard filters as WHERE conditions — returns raw candidates in price-ascending order.
Preference ranking and penalty demotion are applied AFTER this call by `EvidenceAssembler`,
not inside the SQL query.
```sql
SELECT * FROM articles
WHERE colour_group_name      = $1    -- exact match (if provided)
  AND product_type_name      = $2    -- exact match (if provided)
  AND graphical_appearance_name = $3 -- exact match (if provided)
  AND avg_price              <= $4   -- range (if provided)
  AND article_id             != ALL($5::bigint[])  -- exclude rejected
ORDER BY avg_price ASC
LIMIT 20
```
**Note:** Penalties are NOT in the SQL. They were removed to prevent contradictions
where a user asks for a value (e.g. "White") that is also in their dislike history,
which would produce SQL like `colour = 'White' AND colour != ALL(['White', ...])` → 0 results.

**`get_articles_for_comparison(article_id_a, article_id_b)`**
Used by: `item_compare`
Wraps `get_articles_by_ids()` and returns `(item_a, item_b)` tuple.

### Preference Ranking — `_rank_by_preferences()`

**Location:** `text_rag/core/evidence_assembler.py` (module-level function)

Applied **separately** to Qdrant results and PostgreSQL results before merging.
Both result sets are scored on the same scale so that Qdrant results (placed first in
the merge) are also preference-ranked, not just raw cosine-similarity order.

```
score per article =
    + preference_boost_weight      for each liked attribute match
    + 0.3 × (1 - rank/n)          if colour matches top purchase history colours
    + 0.2 × (1 - rank/n)          if product type matches top purchase history types
    + 0.25                         if index_group matches inferred_gender group
    - 0.5                          for each disliked attribute match (penalties)
                                   ← SKIPPED if the penalised value matches the
                                      current turn's hard filter value
```

**Penalty suppression rule (filter suppression):**
If the user asked for `colour_group_name = "Blue"` this turn, and "Blue" is also in
their dislike penalties, the `-0.5` deduction is suppressed for that attribute.
This prevents the system from demoting the exact thing the user just asked for.

**Penalties only apply to PostgreSQL results.** Qdrant results are ranked by preference
boosts and purchase history only — penalty support for Qdrant is a future addition.

**inferred_gender is a soft boost, not a filter.** A `mixed` gender history or a gift
shopper should still see all results — the `+0.25` bonus is applied to the matching
gender group but nothing is excluded.

### Data Loading (done once at startup)

```
sample_articles.csv   (41,794 article rows)
sample_transactions.csv (transaction history → avg_price per article)
     ↓
Compute avg_price per article_id
     ↓
INSERT INTO articles (...) ON CONFLICT (article_id) DO NOTHING
     ↓
41,794 rows loaded
```

If `SELECT COUNT(*) FROM articles > 0`, loading is skipped on restart.

---

## 8. How Qdrant and PostgreSQL Work Together

Both databases store the same 41,794 H&M articles but serve entirely different
roles. Neither can replace the other.

| If only Qdrant | If only PostgreSQL |
|----------------|--------------------|
| Cannot do full structured queries for article detail | Cannot do semantic search — only exact keyword match |
| No SQL filtering (price BETWEEN, NOT IN list) | `WHERE prod_name LIKE '%jacket%'` misses "blazer", "coat" — no semantic understanding |
| Payload can become stale if articles change | No vector similarity — cannot match meaning |

**Qdrant finds which articles are relevant. PostgreSQL fetches what those articles contain.**

### Combined Flow for INITIAL_REQUEST / REFINEMENT

Both databases run **in parallel** — Qdrant for semantic relevance, PostgreSQL for structured
filtering. Results from each are preference-ranked separately, then merged (Qdrant first).

```
User: "I want a blue casual jacket under £50"
     ↓
Memory Pipeline extracts:
  filters            = {product_type_name: "Jacket", colour_group_name: "Blue", price_max: 50.0}
  preference_boosts  = [{colour: Black, weight: 0.68}, {type: Dress, weight: 0.61}, ...]
  purchase_hints     = {top_colours: [Black, White, ...], inferred_gender: female, ...}
  penalties          = {colour_group_name: ["Orange"]}   ← only strong dislikes (≥ 0.5)
  exclude_ids        = [previously rejected article IDs]
     ↓
┌─────────────────────────────────────┐  ┌────────────────────────────────────────────┐
│ Qdrant semantic_search()            │  │ PostgreSQL search_articles_filtered()      │
│  query  = "blue casual jacket"      │  │  filters = {colour: Blue, type: Jacket,    │
│  filters = {colour, type, price}    │  │            price_max: 50.0}               │
│  exclude = [rejected_ids]           │  │  exclude = [rejected_ids]                  │
│  top_k  = 20                        │  │  limit   = 20                              │
│  → 20 articles (cosine ranked)      │  │  → up to 20 articles (price-ascending)     │
└────────────────┬────────────────────┘  └──────────────────┬─────────────────────────┘
                 │                                           │
                 ↓                                           ↓
  _rank_by_preferences(                      _rank_by_preferences(
    articles = qdrant_results,                 articles = pg_results,
    preference_boosts = [...],                 preference_boosts = [...],
    purchase_hints = {...},                    purchase_hints = {...},
    penalties = None          ← not applied    penalties = {"colour": ["Orange"]},
  )                                            filters = {colour: Blue, type: Jacket}
                                             )
                                             ← "Blue" penalty suppressed because
                                                colour_group_name filter = "Blue"
                 │                                           │
                 └──────────────┬────────────────────────────┘
                                ↓
              Merge: Qdrant results first (preserve semantic priority)
              then PostgreSQL results not already seen (add structural diversity)
              Deduplicate by article_id
                                ↓
              Post-merge hard exclude filter (safety gate for rejected IDs)
                                ↓
              If 0 results → filter relaxation fallback:
                Pass 1: drop price constraints → retry Qdrant + PostgreSQL
                Pass 2: keep only product_type_name → retry Qdrant + PostgreSQL
                        (each fallback also applies _rank_by_preferences to PG results)
                                ↓
              Select top N items (colour diversity applied if N > 2)
                                ↓
EvidenceAssembler builds evidence bundle:
  { action: "catalog_search", items: [N full article dicts] }
     ↓
ResponseGenerator builds prompt → Groq LLM → response
     ↓
HallucinationChecker verifies name/colour/price in response match evidence
     ↓
ContradictionDetector updates session graph with N product nodes
     ↓
Final response → user
```

### Which Database Is Used Per Action

| Action | Qdrant | PostgreSQL |
|--------|--------|------------|
| `catalog_search` | Yes — semantic search for candidates | Yes — fetch full data for candidate IDs |
| `item_detail_lookup` | No | Yes — `get_article_by_id()` |
| `item_attribute_lookup` | No | Yes — `get_article_by_id()` |
| `item_compare` | No | Yes — `get_articles_for_comparison()` |
| `explanation_generate` | No | Yes — `get_article_by_id()` |
| `no_retrieval` | No | No |

---

## 9. Output Structure

Every call to `TextRAGPipeline.process()` returns:

```python
{
    "response_text":       str,    # the response shown to the user
    "hallucination_flag":  bool,   # True if unresolved hallucination after 3 attempts
    "hallucination_score": float,  # 0.0–1.0 severity score
    "flagged_sentences":   list,   # sentences that failed NLI check
    "attempt_count":       int,    # how many generation attempts were made
    "contradiction_found": bool,   # True if cross-turn contradiction was corrected
    "contradiction_count": int,    # number of contradictions corrected
    "contradictions":      list,   # details of each contradiction
    "action":              str,    # which action was triggered
    "items_recommended":   list,   # for catalog_search: items returned
    "product_ids":         list,   # article IDs mentioned in response
    "product_names":       list,   # article names mentioned in response
}
```

---

## 10. Configuration

All settings are in `text_rag/config.py` and can be overridden via environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `articles` | Qdrant collection name |
| `QDRANT_VECTOR_SIZE` | `384` | MiniLM output dimension |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `sunlytics` | Database name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM for response generation |
| `NLI_CONTRADICTION_THRESHOLD` | `0.65` | DeBERTa threshold for hallucination |
| `MAX_REGENERATION_ATTEMPTS` | `3` | Max hallucination retry attempts |
| `MAX_RECOMMENDATIONS` | `2` | Items recommended per turn |
| `PRICE_SCALE` | `595.08` | Multiplier to convert normalised price to £ |
