import os
import json
from groq import Groq
from dotenv import load_dotenv, find_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
load_dotenv(find_dotenv())

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client failed to initialize in Router. {e}")
    client = None

# Lazy MongoDB Connection
_mongo_client = None

async def get_mongo_db():
    global _mongo_client
    if _mongo_client is None:
        mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        _mongo_client = AsyncIOMotorClient(mongo_url)
    return _mongo_client[os.getenv("MONGODB_DB_NAME", "sunlytics_crs")]

async def _agentic_state_recovery(user_message: str, session_id: str, missing_fields: list) -> dict:
    """
    AGENTIC HEALING: Queries MongoDB for the last recommended items in this session,
    and asks the LLM to deduce the missing article IDs based on the user's message.
    """
    print(f"\n[M1 AGENT] 🛠️ Missing payload fields detected: {missing_fields}")
    print(f"[M1 AGENT] 🔍 Querying MongoDB for session '{session_id}' history...")
    
    try:
        db = await get_mongo_db()
        # Look in m1_recommendations first, fallback to standard recommendations
        latest_rec = await db.m1_recommendations.find_one({"session_id": session_id}, sort=[("created_at", -1)])
        if not latest_rec:
            latest_rec = await db.recommendations.find_one({"session_id": session_id}, sort=[("created_at", -1)])
            
        if not latest_rec or "items_recommended" not in latest_rec:
            print("[M1 AGENT] ❌ No previous recommendations found in DB. Cannot heal.")
            return {}

        # Format the recent items to show the LLM
        recent_items = []
        for item in latest_rec["items_recommended"][:5]: # Top 5 recent items
            recent_items.append({
                "article_id": str(item.get("article_id")),
                "name": item.get("name") or item.get("prod_name", ""),
                "colour": item.get("colour") or item.get("colour_group_name", ""),
                "type": item.get("type") or item.get("product_type_name", "")
            })
            
        print(f"[M1 AGENT] 🧠 Found {len(recent_items)} recent items. Asking LLM to deduce IDs...")
        
        prompt = f"""You are a JSON data recovery agent.
        The user said: "{user_message}"
        
        Here are the items recently shown to the user in their UI:
        {json.dumps(recent_items, indent=2)}
        
        You need to identify the correct 'article_id' for the following missing fields: {missing_fields}.
        If the user says "compare the red one and the blue one", find the ID for the red item and the blue item.
        If the user says "tell me about the first one", grab the ID of the first item in the list.
        
        Respond ONLY with a valid JSON object containing exactly the missing fields and their resolved string values.
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        recovered_data = json.loads(response.choices[0].message.content)
        print(f"[M1 AGENT] ✨ Successfully recovered data: {recovered_data}\n")
        return recovered_data

    except Exception as e:
        print(f"[M1 AGENT] ⚠️ State recovery failed: {e}")
        return {}

async def heal_and_route_input(retrieval_input: dict, memory_context: dict) -> dict:
    """
    1. Fixes M3's Classification Action.
    2. Checks if required IDs are missing.
    3. Triggers Agentic DB Recovery only if necessary.
    """
    if not retrieval_input or not client:
        return retrieval_input

    user_message = str(retrieval_input.get("user_message", "")).strip()
    current_action = retrieval_input.get("action")
    
    # --- 1. INTENT CORRECTION ---
    system_prompt = """Categorize the user's message into EXACTLY ONE of these specific actions:
    1. "catalog_search": Looking for new items.
    2. "item_compare": Comparing two or more items (e.g., "compare", "difference", "which is better").
    3. "explanation_generate": Asking WHY an item was recommended.
    4. "item_attribute_lookup": Asking a specific trait (e.g., "what color", "what material").
    5. "item_detail_lookup": General summary (e.g., "tell me more about it").
    6. "NO_RETRIEVAL": Pure chitchat.
    Return JSON format: {"action": "..."}
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Message: {user_message}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        new_action = json.loads(response.choices[0].message.content).get("action")
        if new_action and new_action != current_action:
            print(f"\n[M1 SEMANTIC ROUTER] Corrected Intent: {current_action} ➔ {new_action}")
            retrieval_input["action"] = new_action
    except Exception:
        new_action = current_action

    # --- 2. CONDITIONAL STATE RECOVERY ---
    action = retrieval_input.get("action")
    payload = retrieval_input.get("payload") or {}
    items_in_ctx = retrieval_input.get("items_in_context") or {}
    session_id = memory_context.get("session_id")
    
    missing_fields = []
    
    if action == "item_compare":
        # Check if A or B is missing
        if not (payload.get("article_id_a") or items_in_ctx.get("item_a")):
            missing_fields.append("article_id_a")
        if not (payload.get("article_id_b") or items_in_ctx.get("item_b")):
            missing_fields.append("article_id_b")
            
    elif action in ["item_detail_lookup", "item_attribute_lookup", "explanation_generate"]:
        # Check if single ID is missing
        if not (payload.get("article_id") or items_in_ctx.get("item_a")):
            missing_fields.append("article_id")

    # If data is missing and we have a session ID, trigger the Agent
    if missing_fields and session_id:
        recovered_data = await _agentic_state_recovery(user_message, session_id, missing_fields)
        # Inject recovered data into payload
        for k, v in recovered_data.items():
            payload[k] = v

    retrieval_input["payload"] = payload
    return retrieval_input