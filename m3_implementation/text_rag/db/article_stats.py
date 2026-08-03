# m3_implementation/text_rag/db/article_stats.py
#
# Offline builder for per-article buying statistics used by PersonalizedRanker.
#
# WHY THIS EXISTS:
#   The ranker needs to answer "how many people bought this, and were they
#   like me?". Raw counts cannot answer that here: the dataset has 250
#   customers over 41,794 articles, so the median article has 3 purchases
#   from 2 unique buyers. Ranking on raw counts would amplify noise
#   (4 buys beating 3 buys is not a signal) and would show the same
#   blockbusters to every user.
#
#   So every statistic in this module is either SHRUNK toward its group mean
#   (popularity) or BACKED OFF to a denser level when support is thin
#   (age distribution). Both are recorded with an explicit support level so
#   the ranker can decide whether a reason is trustworthy enough to show.
#
# TABLES PRODUCED:
#   article_stats  — one row per article (41,794 rows)
#   group_stats    — aggregates at 3 backoff levels:
#                      'type_colour'   product_type_name|colour_group_name  (~2,067)
#                      'garment_group' garment_group_name                   (~21)
#                      'global'        single row
#
# HOW TO RUN:
#   python -m text_rag.db.article_stats --build
#   python -m text_rag.db.article_stats --show 554477008
#
# The build reads the CSVs directly (same source as postgres_client /
# customer_profile_loader) and takes roughly 30-60 seconds.

import argparse
import asyncio
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from text_rag.config import ARTICLES_CSV, TRANSACTIONS_CSV
from text_rag.db.postgres_client import get_pool, close_pool

CUSTOMERS_CSV = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'shared', 'main_data_set', 'sample_customers.csv'
)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Dirichlet pseudo-count for popularity shrinkage. Higher = more shrinkage
# toward the product_type average. At 3.0, an article with 4 buys and one
# with 3 buys land within ~2% of each other, which is what we want: that
# difference is noise, not preference signal.
POPULARITY_ALPHA = 3.0

# Minimum buyers-with-known-age before an article's own age distribution is
# trusted. Below this the ranker backs off to type_colour, then garment_group.
MIN_AGE_SUPPORT_ARTICLE = 5
MIN_AGE_SUPPORT_GROUP   = 20

# Window (in days, counting back from the last transaction in the dataset)
# used to decide whether an article is trending.
TREND_WINDOW_DAYS = 90

AGE_BUCKETS = ("16-25", "26-35", "36-50", "51+")


def age_bucket(age) -> str:
    """Maps a numeric age to its bucket. Returns '' when age is unknown."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return ""
    if a <= 0:
        return ""
    if a < 26:
        return "16-25"
    if a < 36:
        return "26-35"
    if a < 51:
        return "36-50"
    return "51+"


# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_STATS_SQL = """
CREATE TABLE IF NOT EXISTS article_stats (
    article_id          BIGINT PRIMARY KEY,
    product_code        INTEGER,
    product_type_name   VARCHAR(30),
    colour_group_name   VARCHAR(20),
    garment_group_name  VARCHAR(35),

    buy_count           INTEGER NOT NULL DEFAULT 0,
    unique_buyers       INTEGER NOT NULL DEFAULT 0,
    repeat_buyers       INTEGER NOT NULL DEFAULT 0,
    repeat_rate         REAL    NOT NULL DEFAULT 0,

    age_16_25           INTEGER NOT NULL DEFAULT 0,
    age_26_35           INTEGER NOT NULL DEFAULT 0,
    age_36_50           INTEGER NOT NULL DEFAULT 0,
    age_51_plus         INTEGER NOT NULL DEFAULT 0,
    age_known           INTEGER NOT NULL DEFAULT 0,

    recent_count        INTEGER NOT NULL DEFAULT 0,
    prior_count         INTEGER NOT NULL DEFAULT 0,
    trend_share         REAL    NOT NULL DEFAULT 0,

    first_sold          DATE,
    last_sold           DATE,

    popularity_lift     REAL    NOT NULL DEFAULT 1.0,
    popularity_pct      REAL    NOT NULL DEFAULT 50.0
);

CREATE INDEX IF NOT EXISTS idx_stats_pcode ON article_stats(product_code);
CREATE INDEX IF NOT EXISTS idx_stats_pop   ON article_stats(popularity_pct);

CREATE TABLE IF NOT EXISTS group_stats (
    level          VARCHAR(20)  NOT NULL,
    group_key      VARCHAR(120) NOT NULL,
    buy_count      INTEGER NOT NULL DEFAULT 0,
    unique_buyers  INTEGER NOT NULL DEFAULT 0,
    n_articles     INTEGER NOT NULL DEFAULT 0,
    age_16_25      INTEGER NOT NULL DEFAULT 0,
    age_26_35      INTEGER NOT NULL DEFAULT 0,
    age_36_50      INTEGER NOT NULL DEFAULT 0,
    age_51_plus    INTEGER NOT NULL DEFAULT 0,
    age_known      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (level, group_key)
);
"""


# ── CSV loading ───────────────────────────────────────────────────────────────

def _load_sources():
    """Loads customers, articles and transactions needed for the aggregation."""
    print("[Stats] Loading customers...")
    cust_age = {}
    with open(CUSTOMERS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cust_age[row['customer_id']] = age_bucket(row.get('age'))
    print(f"[Stats]   {len(cust_age)} customers")

    print("[Stats] Loading articles...")
    articles = {}
    with open(ARTICLES_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            articles[row['article_id']] = {
                'product_code':       row.get('product_code'),
                'product_type_name':  row.get('product_type_name') or '',
                'colour_group_name':  row.get('colour_group_name') or '',
                'garment_group_name': row.get('garment_group_name') or '',
            }
    print(f"[Stats]   {len(articles)} articles")

    print("[Stats] Loading transactions...")
    txns = []
    with open(TRANSACTIONS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            txns.append((row['article_id'], row['customer_id'], row['t_dat']))
    print(f"[Stats]   {len(txns)} transactions")

    return cust_age, articles, txns


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(cust_age: dict, articles: dict, txns: list):
    """
    Builds the per-article and per-group aggregates.

    Returns (article_rows, group_rows).
    """
    # Dataset end date drives the trend window — never hardcode "today",
    # the H&M sample ends in 2020.
    all_dates = sorted({t[2] for t in txns if t[2]})
    last_date = datetime.strptime(all_dates[-1], '%Y-%m-%d').date()
    trend_cut = last_date - timedelta(days=TREND_WINDOW_DAYS)
    prior_cut = trend_cut - timedelta(days=TREND_WINDOW_DAYS)
    print(f"[Stats] Dataset ends {last_date}; "
          f"trend window = last {TREND_WINDOW_DAYS} days (from {trend_cut})")

    buy_count     = Counter()
    buyer_txns    = defaultdict(Counter)          # article -> {customer: n}
    age_counts    = defaultdict(Counter)          # article -> {bucket: n}
    recent_count  = Counter()
    prior_count   = Counter()
    first_sold    = {}
    last_sold     = {}

    for aid, cid, tdat in txns:
        buy_count[aid]         += 1
        buyer_txns[aid][cid]   += 1

        bucket = cust_age.get(cid, "")
        if bucket:
            age_counts[aid][bucket] += 1

        if tdat:
            d = datetime.strptime(tdat, '%Y-%m-%d').date()
            if d > trend_cut:
                recent_count[aid] += 1
            elif d > prior_cut:
                prior_count[aid] += 1
            if aid not in first_sold or d < first_sold[aid]:
                first_sold[aid] = d
            if aid not in last_sold or d > last_sold[aid]:
                last_sold[aid] = d

    # ── Popularity: Dirichlet-shrunk share within product_type ────────────────
    # An article's share of its product_type's transactions, smoothed by
    # POPULARITY_ALPHA pseudo-counts spread over every article in the type.
    # Expressed as a LIFT relative to the type average, so 1.0 = typical,
    # 2.0 = twice the average sales for that product type.
    type_txns     = Counter()
    type_articles = Counter()
    for aid, meta in articles.items():
        ptype = meta['product_type_name'] or 'Unknown'
        type_txns[ptype]     += buy_count.get(aid, 0)
        type_articles[ptype] += 1

    popularity_lift = {}
    for aid, meta in articles.items():
        ptype = meta['product_type_name'] or 'Unknown'
        n     = buy_count.get(aid, 0)
        N     = type_txns[ptype]
        K     = max(type_articles[ptype], 1)
        # Smoothed share; uniform prior share is 1/K, so dividing gives lift.
        shrunk_share = (n + POPULARITY_ALPHA) / (N + POPULARITY_ALPHA * K) if (N + POPULARITY_ALPHA * K) > 0 else 0.0
        popularity_lift[aid] = round(shrunk_share * K, 4)

    # Percentile rank of the lift within the product_type — gives the ranker a
    # bounded 0-100 signal that is comparable across product types.
    by_type = defaultdict(list)
    for aid, meta in articles.items():
        by_type[meta['product_type_name'] or 'Unknown'].append(popularity_lift[aid])
    type_sorted = {t: sorted(v) for t, v in by_type.items()}

    def _percentile_of(ptype, value):
        arr = type_sorted.get(ptype, [])
        if not arr:
            return 50.0
        # Fraction of articles in this type scoring at or below `value`.
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        return round(100.0 * lo / len(arr), 2)

    # ── Assemble article rows ────────────────────────────────────────────────
    article_rows = []
    for aid, meta in articles.items():
        buyers  = buyer_txns.get(aid, {})
        uniq    = len(buyers)
        repeats = sum(1 for n in buyers.values() if n > 1)
        ages    = age_counts.get(aid, Counter())
        known   = sum(ages.values())
        rec     = recent_count.get(aid, 0)
        pri     = prior_count.get(aid, 0)
        ptype   = meta['product_type_name'] or 'Unknown'

        try:
            pcode = int(float(meta['product_code'])) if meta['product_code'] else None
        except (TypeError, ValueError):
            pcode = None

        article_rows.append((
            int(aid),
            pcode,
            meta['product_type_name'][:30]  or None,
            meta['colour_group_name'][:20]  or None,
            meta['garment_group_name'][:35] or None,
            buy_count.get(aid, 0),
            uniq,
            repeats,
            round(repeats / uniq, 4) if uniq else 0.0,
            ages.get("16-25", 0),
            ages.get("26-35", 0),
            ages.get("36-50", 0),
            ages.get("51+",   0),
            known,
            rec,
            pri,
            round(rec / (rec + pri), 4) if (rec + pri) else 0.0,
            first_sold.get(aid),
            last_sold.get(aid),
            popularity_lift[aid],
            _percentile_of(ptype, popularity_lift[aid]),
        ))

    # ── Group aggregates for age backoff ─────────────────────────────────────
    levels = {
        'type_colour':   defaultdict(lambda: {'buys': 0, 'buyers': set(), 'arts': set(), 'ages': Counter()}),
        'garment_group': defaultdict(lambda: {'buys': 0, 'buyers': set(), 'arts': set(), 'ages': Counter()}),
        'global':        defaultdict(lambda: {'buys': 0, 'buyers': set(), 'arts': set(), 'ages': Counter()}),
    }

    for aid, cid, _tdat in txns:
        meta = articles.get(aid)
        if not meta:
            continue
        bucket = cust_age.get(cid, "")
        keys = {
            'type_colour':   f"{meta['product_type_name']}|{meta['colour_group_name']}",
            'garment_group': meta['garment_group_name'] or 'Unknown',
            'global':        'ALL',
        }
        for level, key in keys.items():
            g = levels[level][key]
            g['buys'] += 1
            g['buyers'].add(cid)
            g['arts'].add(aid)
            if bucket:
                g['ages'][bucket] += 1

    group_rows = []
    for level, groups in levels.items():
        for key, g in groups.items():
            ages = g['ages']
            group_rows.append((
                level,
                key[:120],
                g['buys'],
                len(g['buyers']),
                len(g['arts']),
                ages.get("16-25", 0),
                ages.get("26-35", 0),
                ages.get("36-50", 0),
                ages.get("51+",   0),
                sum(ages.values()),
            ))

    return article_rows, group_rows


# ── Build ─────────────────────────────────────────────────────────────────────

async def build(force: bool = False):
    """Creates the tables and populates them from the CSVs."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATE_STATS_SQL)
        print("[Stats] Schema created/verified.")

        existing = await conn.fetchval("SELECT COUNT(*) FROM article_stats")
        if existing > 0 and not force:
            print(f"[Stats] Already built: {existing} rows. Use --force to rebuild.")
            return existing

    cust_age, articles, txns = _load_sources()
    article_rows, group_rows = _aggregate(cust_age, articles, txns)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("TRUNCATE TABLE article_stats")
            await conn.execute("TRUNCATE TABLE group_stats")
            await conn.executemany(
                """INSERT INTO article_stats VALUES (
                       $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                       $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21
                   )""",
                article_rows,
            )
            await conn.executemany(
                "INSERT INTO group_stats VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)",
                group_rows,
            )

    print(f"[Stats] Wrote {len(article_rows)} article_stats rows, "
          f"{len(group_rows)} group_stats rows.")
    await _report(article_rows)
    return len(article_rows)


async def _report(article_rows: list):
    """Prints the distributions so the signal strength is visible after a build."""
    lifts = sorted(r[19] for r in article_rows)
    buys  = sorted(r[5]  for r in article_rows)
    uniq  = sorted(r[6]  for r in article_rows)
    aged  = sum(1 for r in article_rows if r[13] >= MIN_AGE_SUPPORT_ARTICLE)

    def pct(arr, p):
        return arr[min(int(len(arr) * p), len(arr) - 1)]

    print("\n[Stats] --- distribution report ---")
    print(f"  buy_count        median={statistics.median(buys):.0f} "
          f"p90={pct(buys, 0.9)} max={buys[-1]}")
    print(f"  unique_buyers    median={statistics.median(uniq):.0f} "
          f"p90={pct(uniq, 0.9)} max={uniq[-1]}")
    print(f"  popularity_lift  p10={pct(lifts,0.1):.2f} median={statistics.median(lifts):.2f} "
          f"p90={pct(lifts,0.9):.2f} max={lifts[-1]:.2f}")
    print(f"  articles with own age support (>={MIN_AGE_SUPPORT_ARTICLE}): "
          f"{aged} ({100*aged/len(article_rows):.0f}%); rest use backoff")


# ── Inspection ────────────────────────────────────────────────────────────────

async def show(article_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM article_stats WHERE article_id = $1", int(article_id)
        )
        if not row:
            print(f"No stats for article {article_id}")
            return
        d = dict(row)
        print(f"\n--- article_stats for {article_id} ---")
        for k, v in d.items():
            print(f"  {k:20s} {v}")

        gkey = f"{d['product_type_name']}|{d['colour_group_name']}"
        grp = await conn.fetchrow(
            "SELECT * FROM group_stats WHERE level='type_colour' AND group_key=$1", gkey
        )
        if grp:
            print(f"\n  backoff group '{gkey}': buys={grp['buy_count']} "
                  f"age_known={grp['age_known']} "
                  f"[{grp['age_16_25']}/{grp['age_26_35']}/{grp['age_36_50']}/{grp['age_51_plus']}]")


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Article buying-stats builder")
    parser.add_argument("--build", action="store_true", help="Build the stats tables")
    parser.add_argument("--force", action="store_true", help="Rebuild even if populated")
    parser.add_argument("--show",  type=str, default=None, help="Show stats for one article_id")
    args = parser.parse_args()

    try:
        if args.show:
            await show(args.show)
        elif args.build or args.force:
            await build(force=args.force)
        else:
            parser.print_help()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
