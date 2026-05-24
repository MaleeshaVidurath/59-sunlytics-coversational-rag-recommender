# m3_implementation/memory/core/contradiction_detector.py
#
# NOVELTY: First CRS contradiction detector that uses the RAG evidence bundle
# as authoritative ground truth stored in a session memory graph (NetworkX).
# Instead of parsing LLM free text with regex, uses LLM-based claim extraction
# (Groq) to extract product values from responses, then compares against
# evidence-anchored graph nodes. DeBERTa NLI used only for confirmation,
# not extraction. This architecture is unique to RAG-based CRS where the
# evidence bundle is already structured and accurate.
#
# PURPOSE:
#   Ensures every bot response is consistent with the evidence used to generate it.
#   Catches LLM drift — cases where the LLM writes the wrong colour, price, or name
#   despite correct evidence being provided in the prompt.
#
# HOW IT WORKS:
#   1. Load session graph from MongoDB (NetworkX DiGraph, persisted across turns)
#   2. Update graph nodes from evidence bundle (PostgreSQL/Qdrant — always accurate)
#   3. Call Groq to extract what name/colour/price the LLM actually wrote
#   4. Compare extracted values against graph node values (evidence truth)
#   5. If mismatch found: confirm with DeBERTa NLI (score > 0.5)
#   6. If confirmed: fix response text, add contradiction edge to graph
#   7. Persist updated graph + log events to MongoDB
#   8. Return corrected response + full report
#
# INTEGRATION:
#   Called from text_rag/core/rag_pipeline.py after hallucination check passes.
#   Only runs for factual actions: catalog_search, item_detail_lookup,
#   item_attribute_lookup, item_compare, explanation_generate.
#
# OUTPUT STRUCTURE (same as before — rag_pipeline.py needs no changes):
#   {
#     "response_text":       str   — corrected response (or original if no contradiction)
#     "contradiction_found": bool
#     "contradiction_count": int
#     "contradictions":      list  — details of each contradiction
#     "claims_stored":       int   — number of products whose evidence was stored
#     "product_ids":         list  — article_ids in this response
#     "product_names":       list  — product names in this response
#   }

import re
import os
import sys
import json
import networkx as nx
import httpx
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from memory.db.mongo import get_db

# ── NLI model (shared singleton with hallucination checker) ───────────────────
_nli_model = None

def _get_nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        print("[ContradictionDetector] NLI model loaded.")
    return _nli_model

# NLI label mapping for cross-encoder/nli-deberta-v3-base:
#   Label 0 = CONTRADICTION  Label 1 = NEUTRAL  Label 2 = ENTAILMENT
# NLI is used only for confirmation after values_contradict() flags a mismatch.
_NLI_CONFIRMATION_THRESHOLD = 0.5


# ── Value normalisation ───────────────────────────────────────────────────────

def normalise_value(val: str) -> str:
    if not val:
        return ""
    val = str(val).lower().strip()
    val = re.sub(r'[-\s]+', ' ', val)
    val = re.sub(r'£\s*', '£', val)
    return val.strip()


def values_contradict(evidence_val: str, extracted_val: str) -> bool:
    """
    Returns True if the extracted value differs from the evidence value.
    Uses float comparison for prices to allow tiny rounding differences.
    """
    ev = normalise_value(evidence_val)
    ex = normalise_value(extracted_val)
    if not ev or not ex:
        return False
    if ev.startswith('£') and ex.startswith('£'):
        try:
            return abs(float(ev[1:]) - float(ex[1:])) > 0.05
        except Exception:
            return ev != ex
    return ev != ex


# ── Session Graph — persistence ───────────────────────────────────────────────

async def _load_graph(session_id: str) -> nx.DiGraph:
    """
    Load session graph from MongoDB.
    Returns an empty DiGraph if this is the first turn of the session.
    """
    db = get_db()
    try:
        doc = await db.session_graphs.find_one({"session_id": session_id})
        if doc and "graph_data" in doc:
            graph = nx.node_link_graph(doc["graph_data"])
            return graph
    except Exception as e:
        print(f"[GRAPH] Failed to load graph for {session_id}: {e}")
    return nx.DiGraph()


async def _save_graph(session_id: str, graph: nx.DiGraph) -> None:
    """Persist session graph to MongoDB for the next turn."""
    db = get_db()
    try:
        graph_data = nx.node_link_data(graph)
        await db.session_graphs.update_one(
            {"session_id": session_id},
            {"$set": {
                "session_id": session_id,
                "graph_data": graph_data,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )
    except Exception as e:
        print(f"[GRAPH] Failed to save graph for {session_id}: {e}")


# ── Evidence → product refs ───────────────────────────────────────────────────

def _item_to_ref(item: dict) -> Optional[dict]:
    """
    Converts one evidence item dict into a normalised product ref.
    Handles field naming differences between catalog_search and context items.
    Returns None if article_id or name is missing.
    """
    aid   = str(item.get("article_id", "")).strip()
    name  = item.get("name") or item.get("prod_name", "")
    colour = item.get("colour_group_name") or item.get("colour", "")
    price  = item.get("price") or item.get("avg_price", "")

    if not aid or not name:
        return None

    if not price:
        price_str = ""
    elif isinstance(price, str) and '£' in price:
        price_str = price.strip()          # already formatted: "£24.76"
    else:
        try:
            price_str = f"£{float(price):.2f}"
        except (ValueError, TypeError):
            price_str = str(price).strip()
    return {
        "article_id": aid,
        "name":       str(name).strip(),
        "colour":     str(colour).strip(),
        "price":      price_str,
    }


def _extract_product_refs(evidence: dict) -> list[dict]:
    """
    Extract product info dicts from the evidence bundle.
    Evidence comes from PostgreSQL/Qdrant and is always accurate ground truth.
    """
    refs   = []
    action = evidence.get("action", "")

    if action == "catalog_search":
        for item in evidence.get("items", []):
            ref = _item_to_ref(item)
            if ref:
                refs.append(ref)

    elif action in ("item_attribute_lookup", "item_detail_lookup",
                    "explanation_generate"):
        article = evidence.get("article") or {}
        ref = _item_to_ref(article)
        if ref:
            refs.append(ref)

    elif action == "item_compare":
        for key in ("item_a", "item_b"):
            item = evidence.get(key) or {}
            ref = _item_to_ref(item)
            if ref:
                refs.append(ref)

    return refs


# ── Session Graph — node management ──────────────────────────────────────────

def _update_graph_nodes(
    graph:        nx.DiGraph,
    product_refs: list[dict],
    turn_id:      str,
    session_id:   str,
) -> None:
    """
    Add or update product nodes from the current turn's evidence.
    Node values are always overwritten with the latest DB evidence — authoritative.
    """
    for ref in product_refs:
        aid = ref["article_id"]
        if graph.has_node(aid):
            attrs = graph.nodes[aid]
            attrs["name"]           = ref["name"]
            attrs["colour"]         = ref["colour"]
            attrs["price"]          = ref["price"]
            attrs["turn_id"]        = turn_id
            attrs["last_seen_turn"] = turn_id
        else:
            graph.add_node(aid, **{
                "name":            ref["name"],
                "colour":          ref["colour"],
                "price":           ref["price"],
                "turn_id":         turn_id,
                "first_seen_turn": turn_id,
                "last_seen_turn":  turn_id,
                "session_id":      session_id,
            })


# ── Groq claim extraction ─────────────────────────────────────────────────────

_GROQ_EXTRACT_PROMPT = """\
You are a claim extractor for a fashion recommender system.

Products shown in this response:
{product_list}

Bot response text:
{response_text}

Extract the name, colour, and price mentioned for each product in the response.
Return ONLY valid JSON with no explanation, no markdown, no extra text.
Format:
{{
  "ARTICLE_ID": {{
    "name": "product name as written in response",
    "colour": "colour as written in response",
    "price": "£XX.XX"
  }}
}}
Only include fields that are explicitly mentioned in the response.
If a product is not mentioned, omit its article_id entirely.\
"""


async def _extract_claims_groq(
    response_text: str,
    product_refs:  list[dict],
) -> dict[str, dict]:
    """
    Calls Groq (llama-3.1-8b-instant) to extract what name/colour/price
    the LLM actually wrote for each product in the response.

    Returns { article_id: { "name": ..., "colour": ..., "price": ... } }
    Returns {} on any failure — errors are logged but never propagated.
    """
    product_list = "\n".join(
        f"- {ref['article_id']}: {ref['name']}"
        for ref in product_refs
    )
    prompt = _GROQ_EXTRACT_PROMPT.format(
        product_list=product_list,
        response_text=response_text[:1500],
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       "llama-3.1-8b-instant",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  300,
                    "temperature": 0.0,
                },
            )
        result  = resp.json()
        content = result["choices"][0]["message"]["content"].strip()
        print(f"[GROQ-EXTRACT] raw: {content[:200]}")

        # Strip markdown fences if Groq wrapped the JSON
        if content.startswith("```"):
            content = re.sub(r'^```[a-z]*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
            content = content.strip()

        extracted = json.loads(content)
        if not isinstance(extracted, dict):
            print("[GROQ-EXTRACT] Warning: parsed JSON is not a dict — skipping")
            return {}

        # Ensure all keys are strings
        extracted = {str(k): v for k, v in extracted.items()}
        print(f"[GROQ-EXTRACT] extracted {len(extracted)} products from response")
        return extracted

    except json.JSONDecodeError as e:
        print(f"[GROQ-EXTRACT] JSON parse failed: {e} — skipping contradiction check")
        return {}
    except Exception as e:
        print(f"[GROQ-EXTRACT] Groq call failed: {e} — skipping contradiction check")
        return {}


# ── NLI confirmation ──────────────────────────────────────────────────────────

def _confirm_with_nli(
    node_name:     str,
    node_colour:   str,
    node_price:    str,
    extracted_val: str,
    attribute:     str,
) -> tuple[bool, float]:
    """
    Confirms a suspected contradiction using DeBERTa NLI.

    Premise  = what the evidence says (ground truth from DB).
    Hypothesis = what the LLM appears to have written.

    Returns (is_contradiction, contradiction_score).
    Returns (False, 0.0) on any failure — never raises.
    """
    try:
        premise = f"The {node_name} is {node_colour} and costs {node_price}."

        if attribute == "colour":
            hypothesis = f"The {node_name} is {extracted_val} in colour."
        elif attribute == "price":
            hypothesis = f"The {node_name} costs {extracted_val}."
        else:
            hypothesis = f"The product is called {extracted_val}."

        nli    = _get_nli()
        scores = nli.predict([(premise, hypothesis)])
        contra_score    = float(scores[0][0])
        is_contradiction = contra_score > _NLI_CONFIRMATION_THRESHOLD
        return is_contradiction, contra_score

    except Exception as e:
        print(f"[NLI] Confirmation failed: {e} — treating as no contradiction")
        return False, 0.0


# ── Response text correction ──────────────────────────────────────────────────

def _fix_response_text(
    response_text:  str,
    wrong_value:    str,
    correct_value:  str,
) -> str:
    """
    Replaces the wrong value with the correct (evidence) value in the response.
    Tries exact match first, then case-insensitive.
    """
    corrected = response_text

    if wrong_value in corrected:
        corrected = corrected.replace(wrong_value, correct_value)
    else:
        pattern   = re.compile(re.escape(wrong_value), re.IGNORECASE)
        corrected = pattern.sub(correct_value, corrected)

    while "  " in corrected:
        corrected = corrected.replace("  ", " ")

    return corrected.strip()


# ── MongoDB contradiction log ─────────────────────────────────────────────────

async def _log_contradiction(
    session_id:      str,
    turn_id:         str,
    article_id:      str,
    article_name:    str,
    attribute:       str,
    evidence_value:  str,
    extracted_value: str,
    nli_score:       float,
) -> None:
    """Writes a contradiction event to db.contradiction_log."""
    db = get_db()
    entry = {
        "session_id":      session_id,
        "turn_id":         turn_id,
        "detected_at":     datetime.now(timezone.utc).isoformat(),
        "article_id":      article_id,
        "article_name":    article_name,
        "attribute":       attribute,
        "evidence_value":  evidence_value,
        "extracted_value": extracted_value,
        "nli_score":       nli_score,
        "resolution":      "response_corrected",
    }
    try:
        await db.contradiction_log.insert_one(entry)
    except Exception as e:
        print(f"[CONTRA] Failed to log contradiction event: {e}")


# ── Main ContradictionDetector class ─────────────────────────────────────────

class ContradictionDetector:
    """
    Evidence-Anchored Session Graph Contradiction Detector.

    Uses a NetworkX DiGraph as session memory. Product nodes hold the
    authoritative evidence values from PostgreSQL/Qdrant. Groq extracts
    what the LLM actually wrote; mismatches are confirmed by DeBERTa NLI
    and corrected in the response before it reaches the user.
    """

    async def check_and_resolve(
        self,
        response_text: str,
        evidence:      dict,
        session_id:    str,
        user_id:       str,
        turn_id:       str,
    ) -> dict:
        """
        Main entry point. Called from rag_pipeline.py after hallucination check.

        Returns same dict structure as before so rag_pipeline.py needs no changes:
        {
          "response_text":       str,
          "contradiction_found": bool,
          "contradiction_count": int,
          "contradictions":      list,
          "claims_stored":       int,
          "product_ids":         list,
          "product_names":       list,
        }
        """
        action = evidence.get("action", "no_retrieval")
        print(f"\n[CONTRA] ━━━ check_and_resolve() called ━━━")
        print(f"[CONTRA] action={action} session={session_id[:12] if session_id else '?'} turn={turn_id}")

        # Only check actions that make factual product claims
        if action not in {
            "catalog_search", "item_attribute_lookup", "item_compare",
            "explanation_generate", "item_detail_lookup",
        }:
            print(f"[CONTRA] SKIP: action={action} not factual")
            return self._no_check_result(response_text)

        try:
            return await self._run_check(
                response_text, evidence, session_id, user_id, turn_id, action
            )
        except Exception as e:
            print(f"[CONTRA] Unexpected error in check_and_resolve: {e}")
            return self._no_check_result(response_text)

    async def _run_check(
        self,
        response_text: str,
        evidence:      dict,
        session_id:    str,
        user_id:       str,
        turn_id:       str,
        action:        str,
    ) -> dict:

        # ── Step 1: Extract product refs from evidence (ground truth) ────────
        product_refs = _extract_product_refs(evidence)
        if not product_refs:
            print(f"[CONTRA] SKIP: no product refs in evidence")
            return self._no_check_result(response_text)

        article_ids   = [r["article_id"] for r in product_refs]
        product_names = [r["name"]       for r in product_refs]

        # ── Step 2: Load and update session graph ────────────────────────────
        graph = await _load_graph(session_id)
        print(f"[GRAPH] nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

        _update_graph_nodes(graph, product_refs, turn_id, session_id)
        print(f"[GRAPH] after update: nodes={graph.number_of_nodes()}")

        # ── Step 3: Extract claims from LLM response via Groq ────────────────
        extracted = await _extract_claims_groq(response_text, product_refs)

        if not extracted:
            print(f"[CONTRA] Groq returned no claims — saving graph, skipping check")
            await _save_graph(session_id, graph)
            return self._build_result(
                response_text, [], len(product_refs), article_ids, product_names,
            )

        # ── Step 4: Compare extracted claims vs graph node values ────────────
        contradictions = []
        corrected_text = response_text

        for article_id, extracted_fields in extracted.items():
            if not graph.has_node(article_id):
                print(f"[CONTRA] {article_id} not in graph — skipping")
                continue

            node        = graph.nodes[article_id]
            node_name   = node.get("name",   "")
            node_colour = node.get("colour", "")
            node_price  = node.get("price",  "")

            print(f"[CONTRA] checking {article_id} ({node_name})")
            print(f"  evidence : colour={node_colour!r}  price={node_price!r}")
            print(f"  extracted: {extracted_fields}")

            for attribute, evidence_val, extracted_val in (
                ("colour", node_colour, extracted_fields.get("colour", "")),
                ("price",  node_price,  extracted_fields.get("price",  "")),
                ("name",   node_name,   extracted_fields.get("name",   "")),
            ):
                if not extracted_val:
                    continue

                if not values_contradict(evidence_val, extracted_val):
                    continue

                print(f"[CONTRA-CANDIDATE] {article_id} | {attribute} | "
                      f"evidence={evidence_val!r} extracted={extracted_val!r}")

                # ── Step 4b: NLI confirmation ─────────────────────────────
                is_contra, nli_score = _confirm_with_nli(
                    node_name=node_name,
                    node_colour=node_colour,
                    node_price=node_price,
                    extracted_val=extracted_val,
                    attribute=attribute,
                )

                if not is_contra:
                    print(f"[CONTRA] NLI not confirmed (score={nli_score:.3f}) — skip")
                    continue

                print(f"[CONTRA-DETECTED] ⚠ {article_id} | attr={attribute} | "
                      f"evidence={evidence_val!r} | extracted={extracted_val!r} | "
                      f"NLI={nli_score:.3f}")

                # ── Step 4c: Fix response text ────────────────────────────
                corrected_text = _fix_response_text(
                    response_text=corrected_text,
                    wrong_value=extracted_val,
                    correct_value=evidence_val,
                )

                # ── Step 4d: Add contradiction edge to graph ──────────────
                contra_node = f"{article_id}_contra_{attribute}_{turn_id}"
                graph.add_node(contra_node, type="contradiction_event",
                               turn_id=turn_id)
                graph.add_edge(
                    article_id, contra_node,
                    attribute=attribute,
                    old_value=extracted_val,
                    new_value=evidence_val,
                    turn_id=turn_id,
                    nli_score=nli_score,
                )

                contradictions.append({
                    "article_id":      article_id,
                    "article_name":    node_name,
                    "attribute":       attribute,
                    "evidence_value":  evidence_val,
                    "extracted_value": extracted_val,
                    "nli_score":       nli_score,
                    "corrected":       True,
                })

                # ── Step 4e: Log to MongoDB ───────────────────────────────
                await _log_contradiction(
                    session_id=session_id,
                    turn_id=turn_id,
                    article_id=article_id,
                    article_name=node_name,
                    attribute=attribute,
                    evidence_value=evidence_val,
                    extracted_value=extracted_val,
                    nli_score=nli_score,
                )

        # ── Step 5: Persist updated graph ────────────────────────────────────
        await _save_graph(session_id, graph)

        contradiction_found = len(contradictions) > 0
        print(f"[CONTRA] result: found={contradiction_found} "
              f"count={len(contradictions)} stored={len(product_refs)}")

        if contradiction_found:
            print(f"[CONTRA] corrected response: {repr(corrected_text[:200])}")
            print(f"[ContradictionDetector] {len(contradictions)} contradiction(s) resolved.")

        return self._build_result(
            corrected_text, contradictions, len(product_refs),
            article_ids, product_names,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _no_check_result(self, response_text: str) -> dict:
        return {
            "response_text":       response_text,
            "contradiction_found": False,
            "contradiction_count": 0,
            "contradictions":      [],
            "claims_stored":       0,
            "product_ids":         [],
            "product_names":       [],
        }

    def _build_result(
        self,
        response_text: str,
        contradictions: list,
        claims_stored:  int,
        product_ids:    list,
        product_names:  list,
    ) -> dict:
        return {
            "response_text":       response_text,
            "contradiction_found": len(contradictions) > 0,
            "contradiction_count": len(contradictions),
            "contradictions":      contradictions,
            "claims_stored":       claims_stored,
            "product_ids":         product_ids,
            "product_names":       product_names,
        }
