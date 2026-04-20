@echo off
echo ==========================================
echo Starting M2 Conversational Recommender
echo ==========================================

echo Starting FastAPI Backend on port 8000...
start cmd /k "set PYTHONIOENCODING=utf-8 && uvicorn app.main:app --reload --port 8000"

echo Starting React Frontend on port 5173...
start cmd /k "cd frontend && npm run dev"

echo Both servers are starting up! 
echo Once they are ready, open your browser to http://localhost:5173
