# m3_implementation/memory/core/customer_profile_loader.py
#
# Loads and stores purchase history profiles for all 250 customers.
#
# WHAT IT DOES:
#   Reads sample_transactions.csv + sample_articles.csv + sample_customers.csv,
#   computes a rich purchase history profile for every customer, and stores
#   it in MongoDB users collection under purchase_history field.
#
# WHEN TO RUN:
#   Once at project startup, or when CSV data changes.
#   Run: python -m memory.core.customer_profile_loader
#
# WHAT IS STORED PER CUSTOMER:
#   - Top 5 colours by purchase frequency (with counts and percentages)
#   - Top 5 product types by purchase frequency
#   - Top 3 garment groups (broader categories)
#   - Top 3 graphical appearance patterns
#   - Gender inference from index_group_name breakdown
#   - Full price statistics (min, max, avg, median, p25, p75, preferred range)
#   - Budget tier (budget / mid / premium)
#   - Purchase date range and recency score
#   - Average monthly purchase frequency
#   - Customer profile info (age, club status, news frequency)
#
# HOW IT IS ACCESSED:
#   user_manager.get_purchase_history(user_id) → full profile dict
#   user_manager.get_purchase_history_hints(user_id) → compact dict for retrieval_input

import os
import sys
import csv
import asyncio
import statistics
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dotenv import load_dotenv
load_dotenv()

PRICE_SCALE = 595.08  # Multiply normalised price by this to get real £

# Paths — relative to project root
BASE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'shared', 'main_data_set'
)
TRANSACTIONS_CSV = os.path.join(BASE_DIR, 'sample_transactions.csv')
ARTICLES_CSV     = os.path.join(BASE_DIR, 'sample_articles.csv')
CUSTOMERS_CSV    = os.path.join(BASE_DIR, 'sample_customers.csv')
TOP250_CSV       = os.path.join(BASE_DIR, 'top_250_user_transaction_counts.csv')


def _load_csv_files():
    """Loads all four CSV files into memory."""
    print("Loading CSV files...")

    customers = {}
    with open(CUSTOMERS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            customers[row['customer_id']] = row
    print(f"  Customers: {len(customers)}")

    articles = {}
    with open(ARTICLES_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            articles[row['article_id']] = row
    print(f"  Articles: {len(articles)}")

    # Group transactions by customer while loading
    by_customer = defaultdict(list)
    with open(TRANSACTIONS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by_customer[row['customer_id']].append(row)
    total_txns = sum(len(v) for v in by_customer.values())
    print(f"  Transactions: {total_txns}")

    top250_ids = set()
    with open(TOP250_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            top250_ids.add(row['customer_id'])
    print(f"  Top 250 customers: {len(top250_ids)}")

    return customers, articles, by_customer, top250_ids


def _percentile(sorted_data: list, pct: float) -> float:
    """Returns the pct-th percentile from a sorted list."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * pct)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _compute_profile(
    customer_id: str,
    customer_info: dict,
    transactions: list,
    articles: dict,
    reference_date: date = None
) -> dict:
    """
    Computes the full purchase history profile for one customer.
    All prices are converted to real £ using PRICE_SCALE.
    """
    if reference_date is None:
        reference_date = date(2020, 9, 30)  # End of dataset

    # Join transactions with articles
    joined = []
    for t in transactions:
        art = articles.get(t['article_id'])
        if art:
            joined.append({**t, **art})

    if not joined:
        return {}

    total = len(joined)

    # ── Colour analysis ───────────────────────────────────────────────────────
    colour_counts = Counter(
        j['colour_group_name'] for j in joined
        if j.get('colour_group_name') and j['colour_group_name'] not in ('', 'Unknown')
    )
    top_colours = [
        {
            "colour": colour,
            "count":  count,
            "pct":    round(count / total * 100, 1),
            "rank":   rank + 1
        }
        for rank, (colour, count) in enumerate(colour_counts.most_common(5))
    ]

    # ── Product type analysis ─────────────────────────────────────────────────
    type_counts = Counter(
        j['product_type_name'] for j in joined
        if j.get('product_type_name') and j['product_type_name'] not in ('', 'Unknown')
    )
    top_product_types = [
        {
            "type":  ptype,
            "count": count,
            "pct":   round(count / total * 100, 1),
            "rank":  rank + 1
        }
        for rank, (ptype, count) in enumerate(type_counts.most_common(5))
    ]

    # ── Garment group analysis ────────────────────────────────────────────────
    garment_counts = Counter(
        j['garment_group_name'] for j in joined
        if j.get('garment_group_name') and j['garment_group_name'] not in ('', 'Unknown')
    )
    top_garment_groups = [
        {
            "group": group,
            "count": count,
            "pct":   round(count / total * 100, 1),
            "rank":  rank + 1
        }
        for rank, (group, count) in enumerate(garment_counts.most_common(3))
    ]

    # ── Graphical appearance (pattern) analysis ───────────────────────────────
    graphical_counts = Counter(
        j['graphical_appearance_name'] for j in joined
        if j.get('graphical_appearance_name') and
           j['graphical_appearance_name'] not in ('', 'Unknown')
    )
    top_graphical_appearances = [
        {
            "pattern": pattern,
            "count":   count,
            "pct":     round(count / total * 100, 1),
            "rank":    rank + 1
        }
        for rank, (pattern, count) in enumerate(graphical_counts.most_common(3))
    ]

    # ── Gender / index group analysis ─────────────────────────────────────────
    index_counts = Counter(
        j['index_group_name'] for j in joined
        if j.get('index_group_name') and j['index_group_name'] not in ('', 'Unknown')
    )
    index_total = sum(index_counts.values())
    index_group_breakdown = {
        group: round(count / index_total * 100, 1)
        for group, count in index_counts.most_common()
    }

    ladies_pct = index_counts.get('Ladieswear', 0) / index_total
    mens_pct   = index_counts.get('Menswear',   0) / index_total
    if ladies_pct >= 0.60:
        inferred_gender = "female"
    elif mens_pct >= 0.60:
        inferred_gender = "male"
    else:
        inferred_gender = "mixed"

    # ── Section analysis (top 3 sections) ────────────────────────────────────
    section_counts = Counter(
        j['section_name'] for j in joined
        if j.get('section_name') and j['section_name'] not in ('', 'Unknown')
    )
    top_sections = [
        {"section": s, "count": c, "pct": round(c/total*100, 1)}
        for s, c in section_counts.most_common(3)
    ]

    # ── Price statistics ──────────────────────────────────────────────────────
    prices = sorted([
        float(j['price']) * PRICE_SCALE
        for j in joined
        if j.get('price') and float(j['price']) > 0
    ])

    if prices:
        avg_price = statistics.mean(prices)
        med_price = statistics.median(prices)
        p25       = _percentile(prices, 0.25)
        p75       = _percentile(prices, 0.75)

        if avg_price < 15:
            budget_tier = "budget"
        elif avg_price < 40:
            budget_tier = "mid"
        else:
            budget_tier = "premium"

        price_stats = {
            "min":             round(min(prices), 2),
            "max":             round(max(prices), 2),
            "avg":             round(avg_price, 2),
            "median":          round(med_price, 2),
            "p25":             round(p25, 2),
            "p75":             round(p75, 2),
            "preferred_range": [round(p25, 2), round(p75, 2)],
            "budget_tier":     budget_tier,
        }
    else:
        price_stats = {
            "min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0,
            "p25": 0.0, "p75": 0.0, "preferred_range": [0.0, 0.0],
            "budget_tier": "unknown"
        }

    # ── Temporal analysis ─────────────────────────────────────────────────────
    dates_str = sorted(
        t['t_dat'] for t in transactions if t.get('t_dat') and t['t_dat']
    )
    if dates_str:
        first_date = datetime.strptime(dates_str[0],  '%Y-%m-%d').date()
        last_date  = datetime.strptime(dates_str[-1], '%Y-%m-%d').date()
        total_days = (last_date - first_date).days + 1
        active_months = max(1, total_days // 30)

        # Recency: how recently did they shop relative to dataset end
        days_since_last = (reference_date - last_date).days
        recency_score = round(max(0.0, 1.0 - (days_since_last / 365)), 3)

        avg_monthly = round(total / active_months, 1)

        purchase_dates = {
            "first":          dates_str[0],
            "last":           dates_str[-1],
            "active_months":  active_months,
            "total_days":     total_days,
        }
    else:
        purchase_dates = {"first": None, "last": None,
                          "active_months": 0, "total_days": 0}
        recency_score  = 0.0
        avg_monthly    = 0.0

    # ── Customer info from customers CSV ──────────────────────────────────────
    try:
        age = int(float(customer_info.get('age', 0) or 0)) or None
    except (ValueError, TypeError):
        age = None

    # ── Assemble final profile ────────────────────────────────────────────────
    return {
        # Volume
        "total_purchases":    total,
        "unique_articles":    len(set(t['article_id'] for t in transactions)),

        # Colour preferences
        "top_colours":        top_colours,
        "dominant_colour":    top_colours[0]["colour"] if top_colours else None,

        # Product type preferences
        "top_product_types":  top_product_types,
        "dominant_product_type": top_product_types[0]["type"] if top_product_types else None,

        # Garment groups
        "top_garment_groups": top_garment_groups,

        # Pattern preferences
        "top_graphical_appearances": top_graphical_appearances,

        # Gender and index groups
        "inferred_gender":        inferred_gender,
        "index_group_breakdown":  index_group_breakdown,

        # Section preferences
        "top_sections": top_sections,

        # Price behaviour
        "price_stats": price_stats,

        # Temporal behaviour
        "purchase_dates":          purchase_dates,
        "recency_score":           recency_score,
        "avg_monthly_purchases":   avg_monthly,

        # Customer profile
        "age":                     age,
        "club_member_status":      customer_info.get('club_member_status', ''),
        "fashion_news_frequency":  customer_info.get('fashion_news_frequency', ''),
        "active":                  customer_info.get('Active', '') == '1.0',

        # Metadata
        "computed_at":  datetime.utcnow().isoformat(),
        "data_version": "v1",
    }


async def load_all_profiles(force_reload: bool = False):
    """
    Computes and stores purchase history profiles for all 250 customers.

    Checks MongoDB first — if a profile already exists and force_reload
    is False, skips that customer. Only computes missing profiles.

    Args:
        force_reload: If True, recomputes all profiles even if they exist.
    """
    from memory.db.mongo import get_db, connect_to_mongodb

    await connect_to_mongodb()
    db = get_db()

    customers, articles, by_customer, top250_ids = _load_csv_files()

    # Check which customers already have profiles
    existing = set()
    if not force_reload:
        cursor = db.users.find(
            {"purchase_history": {"$exists": True}},
            {"customer_id": 1}
        )
        async for doc in cursor:
            if doc.get('customer_id'):
                existing.add(doc['customer_id'])
        print(f"Already have profiles for: {len(existing)} customers")

    to_process = [
        cid for cid in top250_ids
        if cid not in existing or force_reload
    ]
    print(f"Computing profiles for: {len(to_process)} customers")

    success = 0
    skipped = 0
    errors  = 0

    for i, cid in enumerate(to_process, 1):
        txns     = by_customer.get(cid, [])
        cust_info= customers.get(cid, {})

        if not txns:
            skipped += 1
            continue

        try:
            profile = _compute_profile(cid, cust_info, txns, articles)
            if not profile:
                skipped += 1
                continue

            # Store in MongoDB — upsert into users collection
            await db.users.update_one(
                {"customer_id": cid},
                {"$set": {"purchase_history": profile}},
                upsert=False  # only update existing users, not create new ones
            )

            # Also upsert — create user doc if it does not exist yet
            result = await db.users.find_one({"customer_id": cid})
            if not result:
                await db.users.insert_one({
                    "customer_id":          cid,
                    "user_id":              f"user_hist_{cid[:8]}",
                    "purchase_history":     profile,
                    "attribute_preferences":[],
                    "disliked_attributes":  [],
                    "style_profile":        {},
                    "created_at":           datetime.utcnow().isoformat(),
                })
            else:
                await db.users.update_one(
                    {"customer_id": cid},
                    {"$set": {"purchase_history": profile}}
                )

            success += 1

            if i % 25 == 0:
                print(f"  Progress: {i}/{len(to_process)} — "
                      f"{success} ok, {errors} errors")

        except Exception as e:
            errors += 1
            print(f"  ERROR for {cid[:16]}...: {e}")

    print(f"\nDone. {success} profiles stored, "
          f"{skipped} skipped, {errors} errors.")
    return success


def get_purchase_history_hints(profile: dict) -> dict:
    """
    Returns the compact purchase_history_hints dict used in retrieval_input.
    This is what goes into catalog_search payload.purchase_history_hints.

    Args:
        profile: Full purchase_history dict from MongoDB

    Returns compact hints dict.
    """
    if not profile:
        return {
            "top_colours":        [],
            "top_product_types":  [],
            "inferred_gender":    None,
            "budget_tier":        None,
            "preferred_price_range": None,
            "dominant_colour":    None,
            "dominant_type":      None,
        }

    return {
        # Top colours as simple list — for ranking
        "top_colours": [
            c["colour"] for c in profile.get("top_colours", [])
        ],
        # Top product types as simple list — for ranking
        "top_product_types": [
            t["type"] for t in profile.get("top_product_types", [])
        ],
        # Gender — helps filter index_group_name
        "inferred_gender": profile.get("inferred_gender"),
        # Budget tier — helps filter by price range
        "budget_tier": profile.get("price_stats", {}).get("budget_tier"),
        # Preferred price range [min, max] for recommendations
        "preferred_price_range": profile.get(
            "price_stats", {}
        ).get("preferred_range"),
        # Single dominant values for strong signals
        "dominant_colour": profile.get("dominant_colour"),
        "dominant_type":   profile.get("dominant_product_type"),
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Load customer purchase history profiles into MongoDB"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute all profiles even if they already exist"
    )
    parser.add_argument(
        "--show", type=str, default=None,
        help="Show profile for a specific customer_id prefix (first 8 chars)"
    )
    args = parser.parse_args()

    if args.show:
        # Show one profile
        from memory.db.mongo import get_db, connect_to_mongodb
        await connect_to_mongodb()
        db = get_db()

        import json
        doc = await db.users.find_one(
            {"customer_id": {"$regex": f"^{args.show}"}}
        )
        if doc:
            doc.pop('_id', None)
            ph = doc.get('purchase_history', {})
            print(json.dumps(ph, indent=2))
        else:
            print(f"No user found with customer_id starting with: {args.show}")
        return

    # Load all profiles
    count = await load_all_profiles(force_reload=args.force)
    print(f"\n{count} profiles ready in MongoDB.")


if __name__ == "__main__":
    asyncio.run(main())
