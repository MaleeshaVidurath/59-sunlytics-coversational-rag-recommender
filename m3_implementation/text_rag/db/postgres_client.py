# m3_implementation/text_rag/db/postgres_client.py
#
# PostgreSQL client for the Text RAG system.
#
# RESPONSIBILITIES:
#   - Create and maintain the articles table
#   - Load all 41,794 articles from sample_articles.csv
#   - Execute retrieval queries for each action type
#   - Handle all structured filtering (colour, type, price, etc.)
#
# QUERIES PROVIDED:
#   get_article_by_id()          — single article lookup
#   get_articles_by_ids()        — multiple article lookup
#   search_articles_filtered()   — filtered catalog search with ranking
#   get_articles_for_comparison()— two articles side by side

import asyncio
import asyncpg
import csv
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from text_rag.config import (
    POSTGRES_URL, ARTICLES_CSV, TRANSACTIONS_CSV, PRICE_SCALE
)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Returns the connection pool, creating it if needed."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(POSTGRES_URL, min_size=2, max_size=10)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── Schema creation ────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS articles (
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

CREATE INDEX IF NOT EXISTS idx_colour    ON articles(colour_group_name);
CREATE INDEX IF NOT EXISTS idx_type      ON articles(product_type_name);
CREATE INDEX IF NOT EXISTS idx_idx_grp   ON articles(index_group_name);
CREATE INDEX IF NOT EXISTS idx_garment   ON articles(garment_group_name);
CREATE INDEX IF NOT EXISTS idx_graphical ON articles(graphical_appearance_name);
CREATE INDEX IF NOT EXISTS idx_price     ON articles(avg_price);
CREATE INDEX IF NOT EXISTS idx_section   ON articles(section_name);
CREATE INDEX IF NOT EXISTS idx_pgroup    ON articles(product_group_name);
"""


async def create_schema():
    """Creates the articles table and indexes if they do not exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
    print("[PostgreSQL] Schema created/verified.")


# ── Article loader ─────────────────────────────────────────────────────────────

def _load_article_prices() -> dict:
    """Builds a dict of article_id → avg_price from transactions CSV."""
    prices = {}
    with open(TRANSACTIONS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            aid   = row['article_id']
            price = float(row['price']) * PRICE_SCALE
            if aid not in prices:
                prices[aid] = []
            prices[aid].append(price)
    return {aid: round(sum(ps)/len(ps), 2) for aid, ps in prices.items()}


def _safe_int(val, default=None):
    try:
        v = int(float(val)) if val and val.strip() else default
        return None if v == -1 else v
    except (ValueError, TypeError):
        return default


async def load_articles(force_reload: bool = False):
    """
    Loads all articles from sample_articles.csv into PostgreSQL.
    Skips loading if articles already exist (unless force_reload=True).
    Merges avg_price from sample_transactions.csv.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM articles")
        if count > 0 and not force_reload:
            print(f"[PostgreSQL] Articles already loaded: {count} rows. Skipping.")
            return count

        if force_reload:
            await conn.execute("TRUNCATE TABLE articles")
            print("[PostgreSQL] Cleared existing articles.")

    print("[PostgreSQL] Loading article prices from transactions...")
    article_prices = _load_article_prices()
    print(f"[PostgreSQL] Price data for {len(article_prices)} articles.")

    print("[PostgreSQL] Loading articles from CSV...")
    rows = []
    with open(ARTICLES_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row['article_id']),
                _safe_int(row['product_code']),
                row['prod_name'][:50] if row['prod_name'] else None,
                _safe_int(row['product_type_no']),
                row['product_type_name'][:30] if row['product_type_name'] else None,
                row['product_group_name'][:25] if row['product_group_name'] else None,
                _safe_int(row['graphical_appearance_no']),
                row['graphical_appearance_name'][:25] if row['graphical_appearance_name'] else None,
                _safe_int(row['colour_group_code']),
                row['colour_group_name'][:20] if row['colour_group_name'] else None,
                row['perceived_colour_value_name'][:15] if row['perceived_colour_value_name'] else None,
                row['perceived_colour_master_name'][:20] if row['perceived_colour_master_name'] else None,
                _safe_int(row['department_no']),
                row['department_name'][:45] if row['department_name'] else None,
                row['index_code'][:1] if row['index_code'] else None,
                row['index_name'][:35] if row['index_name'] else None,
                _safe_int(row['index_group_no']),
                row['index_group_name'][:15] if row['index_group_name'] else None,
                _safe_int(row['section_no']),
                row['section_name'][:35] if row['section_name'] else None,
                _safe_int(row['garment_group_no']),
                row['garment_group_name'][:35] if row['garment_group_name'] else None,
                row['detail_desc'] if row['detail_desc'] else None,
                article_prices.get(row['article_id']),
            ))

    INSERT_SQL = """
        INSERT INTO articles VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,
            $13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24
        ) ON CONFLICT (article_id) DO NOTHING
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(INSERT_SQL, rows)

    async with pool.acquire() as conn:
        final_count = await conn.fetchval("SELECT COUNT(*) FROM articles")
    print(f"[PostgreSQL] Loaded {final_count} articles.")
    return final_count


# ── Query functions ────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Converts an asyncpg Record to a plain dict."""
    return dict(row)


async def get_article_by_id(article_id: str) -> Optional[dict]:
    """
    Fetches a single article by article_id.
    Returns full dict with all fields, or None if not found.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM articles WHERE article_id = $1",
            int(article_id)
        )
    return _row_to_dict(row) if row else None


async def get_articles_by_ids(article_ids: list[str]) -> list[dict]:
    """
    Fetches multiple articles by their article_ids.
    Preserves the order of the input list.
    """
    if not article_ids:
        return []
    ids = [int(aid) for aid in article_ids]
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM articles WHERE article_id = ANY($1::bigint[])",
            ids
        )
    id_to_row = {str(r['article_id']): _row_to_dict(r) for r in rows}
    return [id_to_row[aid] for aid in article_ids if aid in id_to_row]


async def search_articles_filtered(
    filters: dict,
    exclude_ids: list[str] = None,
    preference_boosts: list[dict] = None,
    purchase_hints: dict = None,
    limit: int = 20
) -> list[dict]:
    """
    Searches articles using hard filters from retrieval_input.payload.filters.
    Returns up to `limit` results ranked by preference boosts and purchase history.

    Args:
        filters:          Hard constraints from payload.filters
                          Keys: colour_group_name, product_type_name,
                                graphical_appearance_name, index_group_name,
                                price_max, price_min
        exclude_ids:      article_ids to exclude
        preference_boosts: Soft ranking weights from payload.preference_boosts
        purchase_hints:   From payload.purchase_history_hints for secondary ranking
        limit:            How many candidates to fetch (Qdrant re-ranks to top 2)

    Returns ranked list of article dicts.
    """
    conditions = []
    params     = []
    p          = 1

    # Hard filter conditions
    if filters.get('colour_group_name'):
        conditions.append(f"colour_group_name = ${p}")
        params.append(filters['colour_group_name'])
        p += 1

    if filters.get('product_type_name'):
        conditions.append(f"product_type_name = ${p}")
        params.append(filters['product_type_name'])
        p += 1

    if filters.get('graphical_appearance_name'):
        conditions.append(f"graphical_appearance_name = ${p}")
        params.append(filters['graphical_appearance_name'])
        p += 1

    if filters.get('index_group_name'):
        conditions.append(f"index_group_name = ${p}")
        params.append(filters['index_group_name'])
        p += 1

    if filters.get('garment_group_name'):
        conditions.append(f"garment_group_name = ${p}")
        params.append(filters['garment_group_name'])
        p += 1

    if filters.get('price_max'):
        conditions.append(f"avg_price <= ${p}")
        params.append(float(filters['price_max']))
        p += 1

    if filters.get('price_min'):
        conditions.append(f"avg_price >= ${p}")
        params.append(float(filters['price_min']))
        p += 1

    # Exclude rejected items
    if exclude_ids:
        ex_ids = [int(x) for x in exclude_ids if x]
        if ex_ids:
            conditions.append(f"article_id != ALL(${p}::bigint[])")
            params.append(ex_ids)
            p += 1

    # Also filter by inferred gender from purchase hints
    if purchase_hints and purchase_hints.get('inferred_gender'):
        gender = purchase_hints['inferred_gender']
        gender_map = {
            'female': ['Ladieswear', 'Divided'],
            'male':   ['Menswear'],
        }
        allowed = gender_map.get(gender)
        if allowed:
            conditions.append(f"index_group_name = ANY(${p}::text[])")
            params.append(allowed)
            p += 1

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    sql = f"""
        SELECT * FROM articles
        {where_clause}
        ORDER BY avg_price ASC
        LIMIT {limit}
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    articles = [_row_to_dict(r) for r in rows]

    # Apply preference boost ranking
    articles = _rank_by_preferences(articles, preference_boosts, purchase_hints)
    return articles


def _rank_by_preferences(
    articles: list[dict],
    preference_boosts: list[dict] = None,
    purchase_hints: dict = None
) -> list[dict]:
    """
    Re-ranks articles by preference boost weights and purchase history hints.
    Higher score = ranked first.
    """
    if not articles:
        return []

    boost_map = {}
    if preference_boosts:
        for boost in preference_boosts:
            key = (boost['attribute'], boost['value'])
            boost_map[key] = boost['weight']

    top_colours = []
    top_types   = []
    if purchase_hints:
        top_colours = purchase_hints.get('top_colours', [])
        top_types   = purchase_hints.get('top_product_types', [])

    def score(art):
        s = 0.0
        # Preference boost scores
        for (attr, val), weight in boost_map.items():
            if art.get(attr) == val:
                s += weight

        # Purchase history secondary boost (lower weight)
        if art.get('colour_group_name') in top_colours:
            idx = top_colours.index(art['colour_group_name'])
            s += 0.3 * (1 - idx / max(len(top_colours), 1))

        if art.get('product_type_name') in top_types:
            idx = top_types.index(art['product_type_name'])
            s += 0.2 * (1 - idx / max(len(top_types), 1))

        return s

    return sorted(articles, key=score, reverse=True)


async def get_articles_for_comparison(
    article_id_a: str,
    article_id_b: str
) -> tuple[Optional[dict], Optional[dict]]:
    """Fetches two articles for comparison. Returns (item_a, item_b)."""
    results = await get_articles_by_ids([article_id_a, article_id_b])
    item_a  = next((r for r in results if str(r['article_id']) == str(article_id_a)), None)
    item_b  = next((r for r in results if str(r['article_id']) == str(article_id_b)), None)
    return item_a, item_b


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="PostgreSQL article loader")
    parser.add_argument("--force", action="store_true", help="Reload all articles")
    args = parser.parse_args()

    await create_schema()
    await load_articles(force_reload=args.force)
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
