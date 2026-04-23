# m3_implementation/api/routers/auth.py
import csv, os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from memory.db.mongo import get_db
from memory.core.user_manager import UserManager

router = APIRouter(prefix="/api/auth", tags=["auth"])

CUSTOMERS_CSV = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 
    'shared', 'main_data_set', 'sample_customers.csv'
)

def _load_customers():
    customers = []
    try:
        with open(CUSTOMERS_CSV, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                customers.append({
                    "customer_id":          row['customer_id'],
                    "short_id":             row['customer_id'][:12] + "...",
                    "club_member_status":   row.get('club_member_status', ''),
                    "fashion_news_frequency": row.get('fashion_news_frequency', ''),
                    "age":                  row.get('age', ''),
                    "active":               row.get('Active', '') == '1.0',
                })
    except Exception as e:
        print(f"[Auth] Could not load customers CSV: {e}")
    return customers

_customers_cache = None

def get_customers_list():
    global _customers_cache
    if _customers_cache is None:
        _customers_cache = _load_customers()
    return _customers_cache


class LoginRequest(BaseModel):
    customer_id: str


@router.get("/customers")
async def list_customers():
    """Returns all 250 customers for the login picker."""
    customers = get_customers_list()
    return {"customers": customers, "total": len(customers)}


@router.post("/login")
async def login(req: LoginRequest):
    """
    Logs in a customer by customer_id.
    Creates or retrieves the user document in MongoDB.
    Returns user_id and customer profile.
    """
    customers = get_customers_list()
    customer  = next(
        (c for c in customers if c['customer_id'] == req.customer_id), None
    )
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    mgr  = UserManager()
    user = await mgr.get_or_create_user(customer_id=req.customer_id)

    # Load purchase history summary
    db  = get_db()
    doc = await db.users.find_one(
        {"user_id": user.user_id},
        {"purchase_history": 1}
    )
    ph = doc.get("purchase_history", {}) if doc else {}

    return {
        "user_id":              user.user_id,
        "customer_id":          req.customer_id,
        "short_id":             req.customer_id[:16] + "...",
        "club_member_status":   customer.get("club_member_status", ""),
        "fashion_news_frequency": customer.get("fashion_news_frequency", ""),
        "age":                  customer.get("age", ""),
        "active":               customer.get("active", False),
        "purchase_summary": {
            "total_purchases":    ph.get("total_purchases", 0),
            "dominant_colour":    ph.get("dominant_colour", ""),
            "dominant_type":      ph.get("dominant_product_type", ""),
            "budget_tier":        ph.get("price_stats", {}).get("budget_tier", ""),
            "inferred_gender":    ph.get("inferred_gender", ""),
        }
    }
