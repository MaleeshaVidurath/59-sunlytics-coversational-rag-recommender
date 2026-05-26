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

from memory.db.mongo import get_db, get_collection_name

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

async def _load_graph(session_id: str, collection_prefix: str = "m3") -> nx.DiGraph:
    """
    Load session graph from MongoDB.
    Returns an empty DiGraph if this is the first turn of the session.
    collection_prefix selects which member's graph collection to use.
    """
    db = get_db()
    coll_name = get_collection_name("session_graphs", collection_prefix)
    try:
        doc = await db[coll_name].find_one({"session_id": session_id})
        if doc and "graph_data" in doc:
            graph = nx.node_link_graph(doc["graph_data"])
            return graph
    except Exception as e:
        print(f"[GRAPH] Failed to load graph for {session_id}: {e}")
    return nx.DiGraph()


async def _save_graph(
    session_id:        str,
    graph:             nx.DiGraph,
    collection_prefix: str = "m3",
) -> None:
    """Persist session graph to MongoDB for the next turn."""
    db = get_db()
    coll_name = get_collection_name("session_graphs", collection_prefix)
    try:
        graph_data = nx.node_link_data(graph)
        await db[coll_name].update_one(
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
    Captures all available fields so every attribute the LLM might mention
    can be checked against DB ground truth.
    Returns None if article_id or name is missing.
    """
    aid           = str(item.get("article_id", "")).strip()
    name          = item.get("name") or item.get("prod_name", "")
    colour        = item.get("colour_group_name") or item.get("colour", "")
    price         = item.get("price") or item.get("avg_price", "")
    product_type  = item.get("product_type_name") or item.get("type", "")
    pattern       = item.get("graphical_appearance_name", "")
    index_group   = item.get("index_group_name", "")
    section       = item.get("section_name", "")
    garment_group = item.get("garment_group_name", "")

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
        "article_id":    aid,
        "name":          str(name).strip(),
        "colour":        str(colour).strip(),
        "price":         price_str,
        "product_type":  str(product_type).strip(),
        "pattern":       str(pattern).strip(),
        "index_group":   str(index_group).strip(),
        "section":       str(section).strip(),
        "garment_group": str(garment_group).strip(),
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
            for field in ("name", "colour", "price", "product_type",
                          "pattern", "index_group", "section", "garment_group"):
                attrs[field] = ref.get(field, "")
            attrs["turn_id"]        = turn_id
            attrs["last_seen_turn"] = turn_id
        else:
            graph.add_node(aid, **{
                "name":            ref["name"],
                "colour":          ref["colour"],
                "price":           ref["price"],
                "product_type":    ref["product_type"],
                "pattern":         ref["pattern"],
                "index_group":     ref["index_group"],
                "section":         ref["section"],
                "garment_group":   ref["garment_group"],
                "turn_id":         turn_id,
                "first_seen_turn": turn_id,
                "last_seen_turn":  turn_id,
                "session_id":      session_id,
            })


# ── Groq claim extraction ─────────────────────────────────────────────────────

_GROQ_EXTRACT_PROMPT = """\
You are a claim extractor for a fashion recommender system.

Products shown in this response (with their correct database values):
{product_list}

Bot response text:
{response_text}

Extract the values the bot wrote for each product.
IMPORTANT: Only extract a field if it is explicitly mentioned in the response text.
If a field is not mentioned in the response, omit it from the output entirely.
Return ONLY valid JSON with no explanation, no markdown, no extra text.

Format:
{{
  "ARTICLE_ID": {{
    "name": "product name as written in response",
    "colour": "colour as written in response",
    "price": "£XX.XX",
    "product_type": "product type as written in response",
    "pattern": "pattern or appearance as written in response",
    "index_group": "index group as written in response",
    "section": "section as written in response",
    "garment_group": "garment group as written in response"
  }}
}}

Only include fields that are explicitly mentioned in the response.
If a product is not mentioned at all, omit its article_id entirely.\
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
    _SHOW_FIELDS = ("colour", "price", "product_type", "pattern",
                    "index_group", "section", "garment_group")
    lines = []
    for ref in product_refs:
        attrs = ", ".join(
            f"{f}={ref[f]}" for f in _SHOW_FIELDS if ref.get(f)
        )
        lines.append(f"- {ref['article_id']}: {ref['name']}\n  [{attrs}]")
    product_list = "\n".join(lines)
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


# All evidence fields stored in graph nodes that Groq may extract and we check.
# Order: most commonly mentioned first so logs are easy to read.
_CHECKABLE_FIELDS = (
    "colour", "price", "name",
    "product_type", "index_group", "section", "garment_group",
)


# ── NLI confirmation ──────────────────────────────────────────────────────────

def _confirm_with_nli(
    node:          dict,
    extracted_val: str,
    attribute:     str,
) -> tuple[bool, float]:
    """
    Confirms a suspected contradiction using DeBERTa NLI.

    Premise   = factual statement built from the graph node (DB ground truth).
    Hypothesis = what the LLM appears to have written for this attribute.

    Returns (is_contradiction, contradiction_score).
    Returns (False, 0.0) on any failure — never raises.
    """
    try:
        name   = node.get("name", "product")
        colour = node.get("colour", "")
        ptype  = node.get("product_type", "")
        price  = node.get("price", "")

        # Build premise from all available node fields
        details = []
        if colour:
            details.append(f"is {colour}")
        if ptype:
            details.append(f"is a {ptype}")
        if price:
            details.append(f"costs {price}")
        premise = (
            f"The {name} " + " and ".join(details) + "."
            if details else f"This is the {name}."
        )

        # Build hypothesis per attribute type
        _hypotheses = {
            "colour":        f"The {name} is {extracted_val} in colour.",
            "price":         f"The {name} costs {extracted_val}.",
            "name":          f"The product is called {extracted_val}.",
            "product_type":  f"The {name} is a {extracted_val}.",
            "pattern":       f"The {name} has a {extracted_val} pattern.",
            "index_group":   f"The {name} is from the {extracted_val} category.",
            "section":       f"The {name} belongs to the {extracted_val} section.",
            "garment_group": f"The {name} is in the {extracted_val} garment group.",
        }
        hypothesis = _hypotheses.get(
            attribute, f"The {name} has {attribute} = {extracted_val}."
        )

        nli    = _get_nli()
        scores = nli.predict([(premise, hypothesis)])
        contra_score     = float(scores[0][0])
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
    session_id:        str,
    turn_id:           str,
    article_id:        str,
    article_name:      str,
    attribute:         str,
    evidence_value:    str,
    extracted_value:   str,
    nli_score:         float,
    collection_prefix: str = "m3",
) -> None:
    """Writes a contradiction event to the appropriate contradiction_log collection."""
    db = get_db()
    coll_name = get_collection_name("contradiction_log", collection_prefix)
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
        "member_model":    collection_prefix,
    }
    try:
        await db[coll_name].insert_one(entry)
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
        response_text:     str,
        evidence:          dict,
        session_id:        str,
        user_id:           str,
        turn_id:           str,
        collection_prefix: str = "m3",
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
                response_text, evidence, session_id, user_id, turn_id, action,
                collection_prefix,
            )
        except Exception as e:
            print(f"[CONTRA] Unexpected error in check_and_resolve: {e}")
            return self._no_check_result(response_text)

    async def _run_check(
        self,
        response_text:     str,
        evidence:          dict,
        session_id:        str,
        user_id:           str,
        turn_id:           str,
        action:            str,
        collection_prefix: str = "m3",
    ) -> dict:

        # ── Step 1: Extract product refs from evidence (ground truth) ────────
        product_refs = _extract_product_refs(evidence)
        if not product_refs:
            print(f"[CONTRA] SKIP: no product refs in evidence")
            return self._no_check_result(response_text)

        article_ids   = [r["article_id"] for r in product_refs]
        product_names = [r["name"]       for r in product_refs]

        # ── Step 2: Load and update session graph ────────────────────────────
        graph = await _load_graph(session_id, collection_prefix)
        print(f"[GRAPH] nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

        _update_graph_nodes(graph, product_refs, turn_id, session_id)
        print(f"[GRAPH] after update: nodes={graph.number_of_nodes()}")

        # ── Step 3: Extract claims from LLM response via Groq ────────────────
        extracted = await _extract_claims_groq(response_text, product_refs)

        if not extracted:
            print(f"[CONTRA] Groq returned no claims — saving graph, skipping check")
            await _save_graph(session_id, graph, collection_prefix)
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

            node      = graph.nodes[article_id]
            node_name = node.get("name", "")

            print(f"[CONTRA] checking {article_id} ({node_name})")
            print(f"  evidence : colour={node.get('colour')!r}  "
                  f"price={node.get('price')!r}  "
                  f"product_type={node.get('product_type')!r}  "
                  f"pattern={node.get('pattern')!r}")
            print(f"  extracted: {extracted_fields}")

            for attribute in _CHECKABLE_FIELDS:
                extracted_val = extracted_fields.get(attribute, "")
                if not extracted_val:
                    continue  # Groq didn't find this field mentioned — skip

                evidence_val = node.get(attribute, "")

                if not values_contradict(evidence_val, extracted_val):
                    continue

                print(f"[CONTRA-CANDIDATE] {article_id} | {attribute} | "
                      f"evidence={evidence_val!r} extracted={extracted_val!r}")

                # ── Step 4b: NLI confirmation ─────────────────────────────
                is_contra, nli_score = _confirm_with_nli(
                    node=node,
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
                    collection_prefix=collection_prefix,
                )

        # ── Step 5: Persist updated graph ────────────────────────────────────
        await _save_graph(session_id, graph, collection_prefix)

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
