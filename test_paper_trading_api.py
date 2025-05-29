from fastapi import FastAPI
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_SESSIONS = [
    {
        "session_id": "session-1",
        "status": "completed",
        "start_time": (datetime.now() - timedelta(hours=2)).isoformat(),
        "end_time": (datetime.now() - timedelta(minutes=30)).isoformat(),
        "config_path": "configs/default_strategy.yaml",
        "duration_minutes": 90,
        "interval_minutes": 1,
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "initial_capital": 10000.0,
        "final_capital": 10250.0,
        "profit_loss": 250.0,
        "profit_loss_pct": 2.5
    },
    {
        "session_id": "session-2",
        "status": "running",
        "start_time": (datetime.now() - timedelta(minutes=45)).isoformat(),
        "end_time": None,
        "config_path": "configs/aggressive_strategy.yaml",
        "duration_minutes": 120,
        "interval_minutes": 1,
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "initial_capital": 10000.0,
        "current_capital": 10120.0,
        "profit_loss": 120.0,
        "profit_loss_pct": 1.2
    }
]

@app.get("/api/paper-trading/sessions")
async def get_paper_trading_sessions():
    return {"sessions": MOCK_SESSIONS}
