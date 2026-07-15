# Latency Logging — Full Process

## Overview

Every conversation turn processed by the chat API is automatically timed and logged to a CSV file. This produces a persistent record of how long each pipeline stage takes per turn, broken down by retrieval tier. The data is used to evaluate the performance impact of the adaptive retrieval strategy.

**Log file location:** `latency_log.csv` (project root)

---

## How Timing Works

### Tool Used — `time.perf_counter()`

Timing uses Python's `time.perf_counter()`, which returns a high-resolution clock value in **seconds with nanosecond precision**. It is the correct tool for measuring short durations (milliseconds) because:
- `time.time()` has lower resolution and can drift
- `time.strftime()` is only for human-readable timestamps, not duration measurement
- `perf_counter()` is monotonic — it never goes backwards

### Three Checkpoints in `chat.py`

Three timestamps are captured around the two main pipeline stages:

```
t_start        ← captured before memory pipeline starts
                      │
                      │  memory.process_turn()
                      │  (intent classification, session lookup,
                      │   context sufficiency evaluation, preference profile)
                      │
t_memory_done  ← captured after memory pipeline finishes
                      │
                      │  rag.process()
                      │  (vector search, structured filter, LLM generation,
                      │   hallucination check, contradiction detection)
                      │
t_rag_done     ← captured after RAG pipeline finishes
```

### Duration Calculations

Durations are computed in milliseconds by subtracting checkpoints and multiplying by 1000:

```
memory_ms = (t_memory_done − t_start)       × 1000
rag_ms    = (t_rag_done   − t_memory_done)  × 1000
total_ms  = (t_rag_done   − t_start)        × 1000
```

| Field | Measures |
|-------|----------|
| `memory_ms` | Memory pipeline only — intent classification, CSE, session/Redis lookup |
| `rag_ms` | RAG pipeline only — retrieval, LLM generation, hallucination + contradiction checks |
| `total_ms` | Full end-to-end turn processing time (= memory_ms + rag_ms) |

---

## Tier Classification

Before logging, `determine_tier()` reads the pipeline output to assign each turn to a retrieval tier and sub-tier. This classification reflects which retrieval path the adaptive system chose for that turn.

```
pipeline_output.retrieval_strategy  +  cse.partial_subtype  +  retrieval_input.exclude_ids
          │
          ▼
    ┌─────────────────────────────────────────────────┐
    │  strategy = NO                                  │  → tier=NO,      sub_tier=—
    │  strategy = FULL,  no exclude_ids               │  → tier=FULL,    sub_tier=STANDARD
    │  strategy = FULL,  has exclude_ids              │  → tier=FULL,    sub_tier=EXCLUSIONS
    │  strategy = PARTIAL, subtype=PARTIAL_RECENT     │  → tier=PARTIAL, sub_tier=RECENT
    │  strategy = PARTIAL, subtype=PARTIAL_SESSION    │  → tier=PARTIAL, sub_tier=SESSION
    └─────────────────────────────────────────────────┘
```

| Tier | Sub-tier | Retrieval behaviour |
|------|----------|---------------------|
| `NO` | `—` | No retrieval — response from memory/LLM only (FEEDBACK, CHITCHAT) |
| `FULL` | `STANDARD` | Full Qdrant vector search across entire catalogue |
| `FULL` | `EXCLUSIONS` | Full search with previously shown items excluded |
| `PARTIAL` | `RECENT` | Redis recent-turn context only — no new catalogue search |
| `PARTIAL` | `SESSION` | MongoDB session history lookup — lightweight re-retrieval |

---

## CSV Writing Process

The logger lives in `api/latency_logger.py`. After RAG finishes, `log_turn()` is called from `chat.py`:

### Step 1 — Header check (`_ensure_header`)
Checks if `latency_log.csv` exists. If the file does not exist yet, it creates it and writes the column header row first. This runs on every turn but only creates the header once.

### Step 2 — Build the row
Assembles 11 fields into a list:

```python
row = [
    time.strftime("%Y-%m-%d %H:%M:%S"),   # human-readable timestamp
    session_id,
    turn_id,
    label,                                 # intent label (e.g. INITIAL_REQUEST)
    tier,                                  # NO / FULL / PARTIAL
    sub_tier,                              # — / STANDARD / EXCLUSIONS / RECENT / SESSION
    user_message,
    round(memory_ms, 1),
    round(rag_ms,    1),
    round(total_ms,  1),
    response_status,                       # OK / HALLUCINATION / CONTRADICTION / HALL+CONTRA / ERROR
]
```

### Step 3 — Append to CSV
The file is opened in **append mode (`"a"`)** so each turn adds exactly one new row without touching existing data. The file is opened and closed on every turn — no in-memory buffer — meaning the log is always readable even if the server crashes mid-session.

```python
with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
```

### Step 4 — Console print
After writing, a summary line is printed to the server console for real-time monitoring:
```
[LATENCY] tier=FULL     sub=STANDARD    memory= 8545ms  rag=16715ms  total=25261ms  status=OK
```

---

## Response Status Field

`compute_response_status()` reads quality flags from the RAG result and produces one of five status values:

| Status | Meaning |
|--------|---------|
| `OK` | Clean response, no issues detected |
| `HALLUCINATION` | Hallucination guard flagged a mismatch — regeneration attempted |
| `CONTRADICTION` | Cross-turn contradiction detected in session |
| `HALL+CONTRA` | Both hallucination and contradiction flagged |
| `ERROR` | RAG result was empty — pipeline exception occurred |

---

## CSV Column Reference

| Column | Type | Example |
|--------|------|---------|
| `timestamp` | string | `2026-05-29 22:26:10` |
| `session_id` | string | `sess_d19bef9a` |
| `turn_id` | string | `turn_7bdb03ef` |
| `label` | string | `INITIAL_REQUEST` |
| `tier` | string | `FULL` |
| `sub_tier` | string | `STANDARD` |
| `user_message` | string | `I need 4 casual shirts` |
| `memory_ms` | float | `8545.7` |
| `rag_ms` | float | `16715.8` |
| `total_ms` | float | `25261.5` |
| `response_status` | string | `OK` |

---

## Sample Rows from Live Data

```
timestamp            session_id    turn_id       label              tier     sub_tier  memory_ms  rag_ms   total_ms  status
2026-05-29 22:26:10  sess_d19bef9a turn_7bdb03ef INITIAL_REQUEST    FULL     STANDARD  8545.7     16715.8  25261.5   OK
2026-05-29 22:29:53  sess_d19bef9a turn_a6e52bae ATTRIBUTE_QUESTION PARTIAL  RECENT    376.2      3212.8   3589.0    OK
2026-05-29 22:32:50  sess_d19bef9a turn_797a6a58 ATTRIBUTE_QUESTION PARTIAL  RECENT    461.2      8924.4   9385.6    CONTRADICTION
2026-05-29 22:38:51  sess_d19bef9a turn_cc8a10e8 SELECTION_REFERENCE PARTIAL RECENT   349.4      3162.1   3511.4    OK
2026-05-29 22:40:17  sess_d19bef9a turn_a62e06d2 EXPLANATION_WHY    NO       —         363.6      14.4     378.0     OK
```

---

## Aggregated Results (summary_stats.csv)

After collecting enough turns, the log was aggregated into per-tier statistics:

| Tier | n | Mean (ms) | Median (ms) | Std Dev | Min (ms) | Max (ms) |
|------|---|-----------|-------------|---------|----------|----------|
| NO | 23 | 865.9 | 888.3 | 599.8 | 98.5 | 1972.3 |
| PARTIAL/RECENT | 24 | 3102.7 | 3116.7 | 684.5 | 1543.8 | 4368.8 |
| PARTIAL/SESSION | 17 | 3179.1 | 3244.0 | 506.4 | 2268.0 | 4185.2 |
| FULL/STANDARD | 16 | 4855.6 | 4516.6 | 891.3 | 3826.6 | 6659.4 |
| FULL/EXCLUSIONS | 19 | 5102.3 | 5028.5 | 1160.8 | 3401.2 | 7664.9 |

**Key finding:** NO retrieval is ~5.6× faster than FULL retrieval, confirming that the adaptive strategy delivers a meaningful performance benefit — turns that do not need retrieval (FEEDBACK, CHITCHAT) are served in under 1 second on average.

---

## Files

| File | Role |
|------|------|
| `m3_implementation/api/latency_logger.py` | Logger — timing, tier classification, CSV write |
| `m3_implementation/api/routers/chat.py` | Captures `t_start`, `t_memory_done`, `t_rag_done` and calls `log_turn()` |
| `latency_log.csv` | Live append-only log — one row per turn |
| `m3_implementation/test_result/adaptive_rag_result/summary_stats.csv` | Aggregated per-tier statistics |
