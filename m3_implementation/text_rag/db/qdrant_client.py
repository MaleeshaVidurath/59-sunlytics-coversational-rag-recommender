# m3_implementation/text_rag/db/qdrant_client.py
#
# Qdrant vector database client for semantic article search.
#
# WHAT IT DOES:
#   - Connects to Qdrant running locally on port 6333
#   - Creates the articles collection with cosine similarity
#   - Indexes all 41,794 articles as vectors using all-MiniLM-L6-v2
#     Text embedded: "prod_name + product_type_name + colour_group_name
#                     + garment_group_name + detail_desc"
#   - Provides semantic_search() for finding relevant articles
#     given a natural language query + hard filters
#
# QDRANT SETUP (Windows):
#   Option A — Docker:
#     docker run -p 6333:6333 qdrant/qdrant
#   Option B — Direct binary:
#     Download from https://github.com/qdrant/qdrant/releases
#     Run: qdrant.exe
#
# Each article stored as:
#   vector:  384-dim embedding of article text
#   payload: all article fields (for filtering without DB join)

import asyncio
import os
import sys
import csv
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from text_rag.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    QDRANT_VECTOR_SIZE, QDRANT_DISTANCE,
    ARTICLES_CSV, TRANSACTIONS_CSV, PRICE_SCALE, EMBEDDING_MODEL
)

_qdrant_client = None
_embed_model   = None


def get_qdrant():
    """Returns Qdrant client, creating it if needed."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"[Qdrant] Connected to {QDRANT_HOST}:{QDRANT_PORT}")
    return _qdrant_client


def get_embed_model():
    """Returns the sentence embedding model, loading it if needed."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[Qdrant] Embedding model loaded: {EMBEDDING_MODEL}")
    return _embed_model


def _make_article_text(article: dict) -> str:
    """
    Creates the text to embed for each article.
    Combines the most semantically meaningful fields.
    """
    parts = [
        article.get('prod_name',                 ''),
        article.get('product_type_name',          ''),
        article.get('colour_group_name',          ''),
        article.get('garment_group_name',         ''),
        article.get('graphical_appearance_name',  ''),
        article.get('index_group_name',           ''),
        article.get('section_name',               ''),
        article.get('detail_desc',                ''),
    ]
    return ' '.join(p for p in parts if p).strip()


# ── Collection management ──────────────────────────────────────────────────────

def create_collection():
    """Creates the Qdrant collection if it does not exist."""
    from qdrant_client.models import Distance, VectorParams
    client = get_qdrant()

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        print(f"[Qdrant] Collection '{QDRANT_COLLECTION}' already exists.")
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=Distance.COSINE,
        )
    )
    print(f"[Qdrant] Created collection '{QDRANT_COLLECTION}'.")


def get_collection_count() -> int:
    """Returns the number of vectors in the collection."""
    client = get_qdrant()
    info   = client.get_collection(QDRANT_COLLECTION)
    return info.points_count or 0


# ── Article indexing ───────────────────────────────────────────────────────────

def _load_prices() -> dict:
    """Loads avg price per article from transactions CSV."""
    prices = {}
    with open(TRANSACTIONS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            aid   = row['article_id']
            price = float(row['price']) * PRICE_SCALE
            if aid not in prices:
                prices[aid] = []
            prices[aid].append(price)
    return {aid: round(sum(ps)/len(ps), 2) for aid, ps in prices.items()}


def index_articles(force_reload: bool = False, batch_size: int = 256):
    """
    Indexes all articles from sample_articles.csv into Qdrant.

    Each article is stored as:
      - vector: embedding of prod_name + type + colour + description
      - payload: all article fields + avg_price (for filtering)

    Args:
        force_reload: Delete existing collection and reindex
        batch_size:   How many articles to encode and upload per batch
    """
    from qdrant_client.models import PointStruct

    if force_reload:
        client = get_qdrant()
        client.delete_collection(QDRANT_COLLECTION)
        print(f"[Qdrant] Deleted existing collection.")
        create_collection()

    current_count = get_collection_count()
    if current_count > 0 and not force_reload:
        print(f"[Qdrant] Already indexed: {current_count} articles. Skipping.")
        return current_count

    print("[Qdrant] Loading prices...")
    article_prices = _load_prices()

    print("[Qdrant] Loading articles...")
    articles = []
    with open(ARTICLES_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            articles.append(row)
    print(f"[Qdrant] {len(articles)} articles to index.")

    model = get_embed_model()
    total_indexed = 0

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]

        # Create texts to embed
        texts = [_make_article_text(a) for a in batch]

        # Encode batch
        vectors = model.encode(texts, show_progress_bar=False).tolist()

        # Build Qdrant points
        points = []
        for art, vector in zip(batch, vectors):
            aid = int(art['article_id'])
            payload = {
                'article_id':               aid,
                'prod_name':                art.get('prod_name', ''),
                'product_type_name':        art.get('product_type_name', ''),
                'product_group_name':       art.get('product_group_name', ''),
                'colour_group_name':        art.get('colour_group_name', ''),
                'graphical_appearance_name':art.get('graphical_appearance_name', ''),
                'perceived_colour_master_name': art.get('perceived_colour_master_name', ''),
                'index_group_name':         art.get('index_group_name', ''),
                'garment_group_name':       art.get('garment_group_name', ''),
                'section_name':             art.get('section_name', ''),
                'department_name':          art.get('department_name', ''),
                'detail_desc':              art.get('detail_desc', ''),
                'avg_price':                article_prices.get(art['article_id']),
            }
            points.append(PointStruct(id=aid, vector=vector, payload=payload))

        get_qdrant().upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        total_indexed += len(batch)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Indexed {total_indexed}/{len(articles)}...")

    print(f"[Qdrant] Indexing complete. {total_indexed} articles indexed.")
    return total_indexed


# ── Semantic search ────────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    filters: dict = None,
    exclude_ids: list[str] = None,
    top_k: int = 10
) -> list[dict]:
    """
    Performs semantic search in Qdrant with optional hard filters.

    Args:
        query:       Natural language search query (user's message or
                     colour + type description)
        filters:     Hard constraints — same format as payload.filters
                     Applied as Qdrant filter conditions BEFORE scoring
        exclude_ids: article_ids to exclude from results
        top_k:       How many results to return

    Returns list of article payload dicts ranked by semantic similarity.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    try:
        from qdrant_client.models import HasId
        has_id_available = True
    except ImportError:
        has_id_available = False

    model     = get_embed_model()
    query_vec = model.encode([query])[0].tolist()

    # Build Qdrant filter
    must_conditions     = []
    must_not_conditions = []

    if filters:
        filter_field_map = {
            'colour_group_name':         'colour_group_name',
            'product_type_name':         'product_type_name',
            'graphical_appearance_name': 'graphical_appearance_name',
            'index_group_name':          'index_group_name',
            'garment_group_name':        'garment_group_name',
        }
        for key, qdrant_field in filter_field_map.items():
            if filters.get(key):
                must_conditions.append(
                    FieldCondition(
                        key=qdrant_field,
                        match=MatchValue(value=filters[key])
                    )
                )
        if filters.get('price_max'):
            must_conditions.append(
                FieldCondition(
                    key='avg_price',
                    range=Range(lte=float(filters['price_max']))
                )
            )
        if filters.get('price_min'):
            must_conditions.append(
                FieldCondition(
                    key='avg_price',
                    range=Range(gte=float(filters['price_min']))
                )
            )

    # Exclude rejected items
    if exclude_ids:
        ex_ids = [int(x) for x in exclude_ids if x]
        if ex_ids and has_id_available:
            try:
                from qdrant_client.models import HasId
                must_not_conditions.append(HasId(has_id=ex_ids))
            except Exception:
                pass

    qdrant_filter = None
    if must_conditions or must_not_conditions:
        qdrant_filter = Filter(
            must     = must_conditions     if must_conditions     else None,
            must_not = must_not_conditions if must_not_conditions else None,
        )

    client = get_qdrant()

    # qdrant-client >= 1.7 uses query_points instead of search
    try:
        from qdrant_client.models import QueryRequest
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points
    except AttributeError:
        # Fallback for older versions
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vec,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

    return [
        {**hit.payload, '_score': hit.score}
        for hit in results
    ]


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qdrant article indexer")
    parser.add_argument("--force",  action="store_true", help="Reindex all articles")
    parser.add_argument("--search", type=str, default=None, help="Test search query")
    args = parser.parse_args()

    if args.search:
        results = semantic_search(args.search, top_k=3)
        for r in results:
            print(f"  [{r.get('_score', 0):.3f}] {r['prod_name']} "
                  f"({r['product_type_name']}, {r['colour_group_name']}) "
                  f"£{r.get('avg_price', 'N/A')}")
    else:
        create_collection()
        index_articles(force_reload=args.force)
