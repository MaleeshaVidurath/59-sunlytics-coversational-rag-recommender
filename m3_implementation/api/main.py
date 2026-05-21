# m3_implementation/api/main.py
# FastAPI application entry point.
#
# START THE SERVER:
#   cd m3_implementation
#   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
#
# FIRST TIME SETUP — add to .env:
#   POSTGRES_HOST=localhost
#   POSTGRES_PORT=5432
#   POSTGRES_DB=sunlytics
#   POSTGRES_USER=postgres
#   POSTGRES_PASSWORD=sunlytics123

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()

from memory.db.mongo  import connect_to_mongodb, close_mongodb_connection
from memory.db.redis_client import connect_to_redis, close_redis_connection
from text_rag.db.postgres_client import create_schema
from text_rag.db.qdrant_client   import get_qdrant

import api.dependencies as deps
from api.routers import auth, chat, sessions
from memory.core.rl_signal_collector import get_rl_collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("[API] Starting up...")

    # Connect all databases
    await connect_to_mongodb()
    await connect_to_redis()
    await create_schema()
    get_qdrant()   # verify Qdrant connection

    # Create RL experiences index (safe to run every startup)
    try:
        from memory.db.mongo import get_db
        db = get_db()
        await db.rl_experiences.create_index("session_id")
        await db.rl_experiences.create_index("reward_source")
        await db.rl_experiences.create_index([("user_id", 1), ("created_at", -1)])
        print("[API] RL experiences indexes created/verified.")
    except Exception as e:
        print(f"[API] RL index creation warning (non-fatal): {e}")

    # Initialise pipelines (loads DistilBERT, NLI models)
    await deps.init_pipelines()

    print("[API] Ready to serve requests.")
    yield

    # Shutdown
    print("[API] Shutting down...")
    await close_mongodb_connection()
    await close_redis_connection()
    from text_rag.db.postgres_client import close_pool
    await close_pool()


app = FastAPI(
    title="Sunlytics CRS API",
    description="Conversational Fashion Recommender System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(sessions.router)

# RL signal collection routes
from memory.core.rl_routes import router as rl_router
app.include_router(rl_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Sunlytics CRS"}


# Serve React build from frontend/dist if it exists
frontend_dist = os.path.join(
    os.path.dirname(__file__), '..', '..', 'frontend', 'dist'
)
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    print(f"[API] Serving React build from {frontend_dist}")
