"""Main entry point for the Market Regime Detection API."""

import logging
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

from api import config
from api.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL) if isinstance(config.LOG_LEVEL, str) else config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include API routes
app.include_router(router)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {config.API_TITLE} v{config.API_VERSION}")
    logger.info(f"Server running at http://{config.API_HOST}:{config.API_PORT}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {config.API_TITLE}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    ) 