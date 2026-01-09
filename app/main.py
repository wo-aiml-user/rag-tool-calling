from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.config import settings
from app.route import setup_routes
from app.logger import setup_logger
import os
import socket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    hostname, worker_id = socket.gethostname(), os.getpid()
    logger.info(f"Worker starting: host={hostname}, pid={worker_id}")
    
    # Startup complete
    logger.info("Application startup complete")
    
    # ── FastAPI runs ────────────────────────────────────────────
    yield
    # ── Shutdown ────────────────────────────────────────────────
    
    logger.info("Application shutting down")


# ─────────────── FastAPI app object ───────────────────────────
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)

setup_logger(settings)
app.state.config = settings

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_routes(app)
