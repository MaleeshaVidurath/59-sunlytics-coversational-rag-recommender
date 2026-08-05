# m3_implementation/memory/core/assertion_ledger.py
#
# The session's Assertion Ledger — a provenance graph of every factual
# statement the system has made to this user, and of every database value
# those statements were grounded in.
#
# WHY A GRAPH, AND WHY THIS SHAPE:
#   The previous session graph stored one node per product with mutable
#   attribute fields, and overwrote those fields from the current turn's
#   evidence before reading them back. The read therefore always returned what
#   had just been written, so no earlier turn could influence any decision —
#   the graph was a pass-through and the check it fed was a second copy of the
#   turn-local hallucination check.
#
#   Here, a VALUE IS A NODE. Every distinct value ever attached to an
#   (article, attribute) slot gets its own node, and the edge carrying it
#   records who said it, on which turn, and against which version of the
#   database. A disagreement is then a structural property of the graph —
#   "this attribute slot has more than one live value node" — which is only
#   answerable if history was kept. That is the work the graph actually does.
#
# NODE TYPES
#   prod:<article_id>                       product identity (no mutable values)
#   attr:<article_id>:<attribute>           one attribute slot of one product
#   val:<article_id>:<attribute>:<norm>     one distinct value of that slot
#   turn:<turn_id>                          one conversational turn
#
# EDGE TYPES
#   prod  -HAS_ATTR->      attr
#   attr  -ASSERTED_AS->   val    carries the assertion history for that value
#   turn  -STATED->        val    which turn put that value on the record
#   val   -CONTRADICTS->   val    two live response values, no DB revision between
#   val   -SUPERSEDED_BY-> val    the database value changed — a legitimate revision
#
# ASSERTION SOURCES
#   "evidence"  the value the database gave us on that turn (ground truth)
#   "response"  the value the assistant actually stated to the user
#
# ASSERTION STATUS
#   "active"     currently stands
#   "deferred"   disagreed with this turn's own evidence — the hallucination
#                guard owns that case, so it is recorded but never counted here
#                and never used as a cross-turn anchor
#   "superseded" was correct when stated, but the database value has since changed
#   "corrected"  was wrong and has been rewritten in the outgoing response

import os
import sys
from datetime import datetime, timezone

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memory.db.mongo import get_db, get_collection_name

# Graphs written by the pre-ledger detector use flat product nodes with mutable
# attributes. They cannot be interpreted under this schema, so a session that
# carries one starts with a fresh ledger rather than a misread one.
LEDGER_SCHEMA_VERSION = 2

# Attributes the ledger tracks. Deliberately the same narrow set the extractor
# can read out of free text — storing taxonomy fields no response ever states
# would fill the graph with slots that can never be checked.
TRACKED_ATTRIBUTES = ("name", "colour", "price", "product_type")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_value(val) -> str:
    """Value-equality key. Mirrors the extractor's normalisation."""
    from text_rag.core.assertion_extractor import normalise_value as _nv
    return _nv(val)


def values_differ(a, b) -> bool:
    from text_rag.core.assertion_extractor import values_differ as _vd
    return _vd(a, b)


# ── Node id helpers ───────────────────────────────────────────────────────────

def prod_node(article_id: str) -> str:
    return f"prod:{article_id}"


def attr_node(article_id: str, attribute: str) -> str:
    return f"attr:{article_id}:{attribute}"


def val_node(article_id: str, attribute: str, value) -> str:
    return f"val:{article_id}:{attribute}:{normalise_value(value)}"


def turn_node(turn_id: str) -> str:
    return f"turn:{turn_id}"


class AssertionLedger:
    """
    Session-scoped provenance graph. One instance per turn: load it, begin the
    turn, feed it evidence and assertions, then save it.
    """

    def __init__(self, graph: nx.DiGraph = None):
        self.graph = graph if graph is not None else nx.DiGraph()
        self.graph.graph.setdefault("schema_version", LEDGER_SCHEMA_VERSION)
        self.graph.graph.setdefault("turn_seq", 0)
        self.turn_id: str = ""
        self.turn_ordinal: int = 0

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    async def load(cls, session_id: str, collection_prefix: str = "m3") -> "AssertionLedger":
        """
        Loads this session's ledger. Returns an empty ledger for a new session,
        for a session whose stored graph predates this schema, or on any read
        failure — the detector then behaves as if this were turn one, which is
        safe rather than wrong.
        """
        db        = get_db()
        coll_name = get_collection_name("session_graphs", collection_prefix)
        try:
            doc = await db[coll_name].find_one({"session_id": session_id})
        except Exception as e:
            print(f"[LEDGER] load failed for {session_id}: {e}")
            return cls()

        if not doc or "graph_data" not in doc:
            return cls()

        if doc.get("schema_version") != LEDGER_SCHEMA_VERSION:
            print(f"[LEDGER] stored graph is schema "
                  f"v{doc.get('schema_version', 1)} — starting a fresh ledger")
            return cls()

        try:
            return cls(nx.node_link_graph(doc["graph_data"]))
        except Exception as e:
            print(f"[LEDGER] graph rebuild failed for {session_id}: {e}")
            return cls()

    async def save(self, session_id: str, collection_prefix: str = "m3") -> None:
        db        = get_db()
        coll_name = get_collection_name("session_graphs", collection_prefix)
        try:
            await db[coll_name].update_one(
                {"session_id": session_id},
                {"$set": {
                    "session_id":     session_id,
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "graph_data":     nx.node_link_data(self.graph),
                    "updated_at":     _now(),
                }},
                upsert=True,
            )
        except Exception as e:
            print(f"[LEDGER] save failed for {session_id}: {e}")

    # ── Turn bookkeeping ──────────────────────────────────────────────────────

    def begin_turn(self, turn_id: str, action: str = "", strategy: str = "") -> int:
        """
        Registers the current turn and returns its 1-based ordinal within the
        session. Ordinals come from the ledger itself rather than from the
        session document, so the comparison "was this said on an earlier turn?"
        never depends on another collection being in step.

        Idempotent: re-running the same turn (a retry) reuses its ordinal.
        """
        self.turn_id = turn_id or f"anon_{self.graph.graph['turn_seq'] + 1}"
        node = turn_node(self.turn_id)

        if self.graph.has_node(node):
            self.turn_ordinal = int(self.graph.nodes[node].get("turn_ordinal", 0))
            return self.turn_ordinal

        self.graph.graph["turn_seq"] = int(self.graph.graph.get("turn_seq", 0)) + 1
        self.turn_ordinal = self.graph.graph["turn_seq"]
        self.graph.add_node(
            node,
            type="turn",
            turn_id=self.turn_id,
            turn_ordinal=self.turn_ordinal,
            action=action,
            strategy=strategy,
            ts=_now(),
        )
        return self.turn_ordinal

    # ── Internal structure builders ───────────────────────────────────────────

    def _ensure_product(self, article_id: str, name_hint: str = "") -> str:
        node = prod_node(article_id)
        if not self.graph.has_node(node):
            self.graph.add_node(
                node,
                type="product",
                article_id=article_id,
                name=name_hint or "",
                first_seen_turn=self.turn_id,
                first_seen_ordinal=self.turn_ordinal,
            )
        elif name_hint and not self.graph.nodes[node].get("name"):
            self.graph.nodes[node]["name"] = name_hint
        self.graph.nodes[node]["last_seen_turn"]    = self.turn_id
        self.graph.nodes[node]["last_seen_ordinal"] = self.turn_ordinal
        return node

    def _ensure_attr(self, article_id: str, attribute: str, name_hint: str = "") -> str:
        p_node = self._ensure_product(article_id, name_hint)
        a_node = attr_node(article_id, attribute)
        if not self.graph.has_node(a_node):
            self.graph.add_node(
                a_node,
                type="attribute",
                article_id=article_id,
                attribute=attribute,
                evidence_value="",
                evidence_version=0,
                evidence_first_turn="",
                evidence_last_turn="",
            )
            self.graph.add_edge(p_node, a_node, rel="HAS_ATTR")
        return a_node

    def _ensure_value(self, article_id: str, attribute: str, value) -> str:
        v_node = val_node(article_id, attribute, value)
        if not self.graph.has_node(v_node):
            self.graph.add_node(
                v_node,
                type="value",
                article_id=article_id,
                attribute=attribute,
                value=str(value),
                norm=normalise_value(value),
            )
        return v_node

    # ── Evidence side ─────────────────────────────────────────────────────────

    def evidence_record(self, article_id: str, attribute: str) -> dict:
        """
        The database value currently on record for this slot, with its version.
        version 0 means the ledger has never seen evidence for this slot.
        """
        a_node = attr_node(article_id, attribute)
        if not self.graph.has_node(a_node):
            return {"value": "", "version": 0, "last_turn": ""}
        n = self.graph.nodes[a_node]
        return {
            "value":     n.get("evidence_value", ""),
            "version":   int(n.get("evidence_version", 0)),
            "last_turn": n.get("evidence_last_turn", ""),
        }

    def apply_evidence(
        self,
        article_id: str,
        attribute:  str,
        value,
        name_hint:  str = "",
    ) -> dict:
        """
        Records the database value for this slot on this turn.

        Returns a revision descriptor when the database value has CHANGED since
        the ledger last saw it, otherwise None. A revision is not an error: the
        earlier statement was true when it was made. The caller uses the
        descriptor to notify the user against the turns that quoted the old
        value, and the ledger marks those assertions "superseded" rather than
        "corrected".
        """
        if not value and value != 0:
            return None

        a_node = self._ensure_attr(article_id, attribute, name_hint)
        n      = self.graph.nodes[a_node]
        prior  = n.get("evidence_value", "")
        version = int(n.get("evidence_version", 0))

        v_node = self._ensure_value(article_id, attribute, value)
        self._append_assertion(
            a_node, v_node,
            source="evidence",
            status="active",
            sentence_idx=None,
            evidence_version=version + (1 if (version == 0 or values_differ(prior, value)) else 0),
        )

        # First time we have ever seen evidence for this slot.
        if version == 0:
            n["evidence_value"]      = str(value)
            n["evidence_version"]    = 1
            n["evidence_first_turn"] = self.turn_id
            n["evidence_last_turn"]  = self.turn_id
            return None

        # Unchanged — just refresh when it was last verified.
        if not values_differ(prior, value):
            n["evidence_last_turn"] = self.turn_id
            return None

        # Changed: a genuine catalogue revision.
        new_version = version + 1
        n["evidence_value"]     = str(value)
        n["evidence_version"]   = new_version
        n["evidence_last_turn"] = self.turn_id

        old_v_node = val_node(article_id, attribute, prior)
        affected   = self._supersede(article_id, attribute, prior)
        if self.graph.has_node(old_v_node):
            self.graph.add_edge(
                old_v_node, v_node,
                rel="SUPERSEDED_BY",
                turn_id=self.turn_id,
                turn_ordinal=self.turn_ordinal,
                ts=_now(),
            )

        print(f"[LEDGER] REVISION {article_id}.{attribute}: "
              f"{prior!r} → {value!r} (v{version}→v{new_version}), "
              f"{len(affected)} earlier turn(s) affected")

        return {
            "article_id":     article_id,
            "attribute":      attribute,
            "old_value":      str(prior),
            "new_value":      str(value),
            "old_version":    version,
            "new_version":    new_version,
            "product_name":   self.graph.nodes[prod_node(article_id)].get("name", ""),
            "affected_turns": affected,
        }

    def _supersede(self, article_id: str, attribute: str, old_value) -> list[dict]:
        """
        Marks every response-sourced assertion of `old_value` as superseded and
        returns the turns that made them, so the user can be told which of the
        assistant's earlier messages is now out of date.
        """
        a_node = attr_node(article_id, attribute)
        v_node = val_node(article_id, attribute, old_value)
        if not (self.graph.has_node(a_node) and self.graph.has_edge(a_node, v_node)):
            return []

        affected = []
        for rec in self.graph.edges[a_node, v_node].get("assertions", []):
            if rec.get("source") != "response":
                continue
            if rec.get("status") == "active":
                rec["status"] = "superseded"
            affected.append({
                "turn_id":      rec.get("turn_id", ""),
                "turn_ordinal": rec.get("turn_ordinal", 0),
            })
        return affected

    # ── Assertion side ────────────────────────────────────────────────────────

    def _append_assertion(
        self,
        a_node: str,
        v_node: str,
        source: str,
        status: str,
        sentence_idx,
        evidence_version: int,
    ) -> None:
        """
        Appends one assertion record to the attr→val edge. A DiGraph holds a
        single edge per node pair, so the history lives as a list on that edge
        rather than as repeated parallel edges — this keeps the graph plainly
        serialisable through node_link_data.
        """
        if not self.graph.has_edge(a_node, v_node):
            self.graph.add_edge(a_node, v_node, rel="ASSERTED_AS", assertions=[])

        self.graph.edges[a_node, v_node].setdefault("assertions", []).append({
            "turn_id":          self.turn_id,
            "turn_ordinal":     self.turn_ordinal,
            "source":           source,
            "status":           status,
            "sentence_idx":     sentence_idx,
            "evidence_version": evidence_version,
            "ts":               _now(),
        })

        t_node = turn_node(self.turn_id)
        if self.graph.has_node(t_node) and not self.graph.has_edge(t_node, v_node):
            self.graph.add_edge(t_node, v_node, rel="STATED", source=source)

    def record_assertion(
        self,
        article_id:   str,
        attribute:    str,
        value,
        status:       str = "active",
        sentence_idx=None,
        name_hint:    str = "",
    ) -> None:
        """Records what the assistant stated about this slot on this turn."""
        if not value:
            return
        a_node = self._ensure_attr(article_id, attribute, name_hint)
        v_node = self._ensure_value(article_id, attribute, value)
        self._append_assertion(
            a_node, v_node,
            source="response",
            status=status,
            sentence_idx=sentence_idx,
            evidence_version=int(self.graph.nodes[a_node].get("evidence_version", 0)),
        )

    def prior_assertion(self, article_id: str, attribute: str) -> dict:
        """
        The most recent value the assistant STATED for this slot on an EARLIER
        turn and which still stands.

        Deferred assertions are excluded: those disagreed with their own turn's
        evidence, were left to the hallucination guard, and must never become
        the anchor a later turn is judged against. Superseded ones are excluded
        too — the catalogue moved on.
        """
        a_node = attr_node(article_id, attribute)
        if not self.graph.has_node(a_node):
            return None

        best = None
        for _, v_node, edata in self.graph.out_edges(a_node, data=True):
            if edata.get("rel") != "ASSERTED_AS":
                continue
            for rec in edata.get("assertions", []):
                if rec.get("source") != "response":
                    continue
                if rec.get("status") != "active":
                    continue
                if int(rec.get("turn_ordinal", 0)) >= self.turn_ordinal:
                    continue  # same turn or later — not history
                if best is None or rec["turn_ordinal"] > best["turn_ordinal"]:
                    best = {
                        "value":            self.graph.nodes[v_node].get("value", ""),
                        "turn_id":          rec.get("turn_id", ""),
                        "turn_ordinal":     int(rec.get("turn_ordinal", 0)),
                        "evidence_version": int(rec.get("evidence_version", 0)),
                    }
        return best

    def mark_contradiction(
        self,
        article_id:   str,
        attribute:    str,
        wrong_value,
        correct_value,
        nli_score:    float,
    ) -> None:
        """
        Links the two conflicting value nodes and retires the wrong one. The
        CONTRADICTS edge is what turns "these two values disagree" from a
        transient comparison into a queryable fact about the session.
        """
        a_node     = self._ensure_attr(article_id, attribute)
        wrong_node = self._ensure_value(article_id, attribute, wrong_value)
        right_node = self._ensure_value(article_id, attribute, correct_value)

        self.graph.add_edge(
            wrong_node, right_node,
            rel="CONTRADICTS",
            turn_id=self.turn_id,
            turn_ordinal=self.turn_ordinal,
            nli_score=float(nli_score),
            resolved_to=str(correct_value),
            ts=_now(),
        )

        if self.graph.has_edge(a_node, wrong_node):
            for rec in self.graph.edges[a_node, wrong_node].get("assertions", []):
                if (rec.get("turn_id") == self.turn_id
                        and rec.get("source") == "response"
                        and rec.get("status") == "active"):
                    rec["status"] = "corrected"

    # ── Read-side helpers ─────────────────────────────────────────────────────

    def known_products(self) -> list[dict]:
        """
        Every product the ledger has evidence for, newest-seen first.

        Used as the candidate set when the current turn's evidence bundle is
        thin or empty — a follow-up question answered from cache still refers
        to products the ledger already knows, and refusing to look at those was
        exactly what stopped the previous detector from ever seeing a cross-turn
        conflict.
        """
        out = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "product":
                continue
            aid  = attrs.get("article_id", "")
            item = {"article_id": aid, "name": attrs.get("name", "")}
            for attribute in TRACKED_ATTRIBUTES:
                item[attribute] = self.evidence_record(aid, attribute)["value"]
            item["_last_ordinal"] = int(attrs.get("last_seen_ordinal", 0))
            out.append(item)
        out.sort(key=lambda d: d["_last_ordinal"], reverse=True)
        for d in out:
            d.pop("_last_ordinal", None)
        return out

    def stats(self) -> dict:
        counts = {"product": 0, "attribute": 0, "value": 0, "turn": 0}
        for _, attrs in self.graph.nodes(data=True):
            t = attrs.get("type", "")
            if t in counts:
                counts[t] += 1
        rels = {}
        for _, _, edata in self.graph.edges(data=True):
            r = edata.get("rel", "?")
            rels[r] = rels.get(r, 0) + 1
        return {"nodes": counts, "edges": rels,
                "turns": int(self.graph.graph.get("turn_seq", 0))}

    def legacy_node_view(self) -> dict:
        """
        Projects the ledger back into the flat {article_id: {field: value}}
        shape the contradiction-evaluation capture hook was written against, so
        the existing test harness keeps working unchanged.
        """
        view = {}
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "product":
                continue
            aid = attrs.get("article_id", "")
            rec = {
                "name":            self.evidence_record(aid, "name")["value"]
                                   or attrs.get("name", ""),
                "colour":          self.evidence_record(aid, "colour")["value"],
                "price":           self.evidence_record(aid, "price")["value"],
                "product_type":    self.evidence_record(aid, "product_type")["value"],
                "first_seen_turn": attrs.get("first_seen_turn", ""),
                "last_seen_turn":  attrs.get("last_seen_turn", ""),
            }
            view[str(aid)] = {k: v for k, v in rec.items() if v}
        return view
