import os
import sys
import asyncio
import json
import logging
from typing import Optional
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Import the LiveKit server SDK for token generation
from livekit import api

# --- LiveKit Configuration ---
LIVEKIT_API_KEY = "API6wPKwaumhN48"
LIVEKIT_API_SECRET = "qUO3kXPWdpfOfsNoTrzxiUxzoNEvz2dx0hqbVFw8zOL"
LIVEKIT_URL = "http://localhost:7880"  # Change to your server URL if needed

# Initialize LiveKit API client
livekit_api = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)

# --- Existing Application Setup ---

# Ensure repo root on path
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import engine entrypoints
try:
    from vc_engine import init_engine, custom_infer
except ImportError as e:
    raise RuntimeError(f"Failed to import from vc_engine.py. Make sure the file is in the root directory. Error: {e}")

# --- Configuration from Environment Variables ---
CONFIG_PATH = os.path.join(ROOT, "configs", "v2", "vc_wrapper.yaml")
GPU = 0
FP16 = True
LOG_LEVEL = "INFO"

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("seedvc.app")

# --- FastAPI App Initialization ---
app = FastAPI(title="Seed-VC v2 Inference and LiveKit Token Service")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Engine Initialization ---
_engine_ready = False
_model_set = None

@app.on_event("startup")
async def startup_event():
    global _engine_ready, _model_set
    logger.info("Starting up and initializing model engine...")

    class ArgsObj:
        fp16 = FP16
        checkpoint_path = None
        config_path = CONFIG_PATH
        gpu = GPU

    try:
        _model_set = init_engine(ArgsObj())
        _engine_ready = True
        logger.info("Model engine initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize model engine: %s", e, exc_info=True)

def get_engine():
    if not _engine_ready:
        raise HTTPException(status_code=503, detail="Model engine is not ready or failed to initialize.")
    return _model_set

# --- LiveKit Token Endpoint ---
@app.get("/get_token")
async def get_livekit_token(
    identity: str = Query("user", description="Participant identity"),
    room: str = Query("testroom", description="LiveKit room name")
):
    """
    Generates a JWT token for a client to connect to a LiveKit room.
    """
    try:
        # Create a grant for the token
        token = (
            livekit_api.with_identity(identity)
            .with_name(identity)  # Optional: display name
            .with_grants(api.VideoGrant(room_join=True, room=room, can_publish=True, can_subscribe=True))
            .to_jwt()
        )
        logger.info("Issued LiveKit token for identity '%s' in room '%s'", identity, room)
        return {"token": token}
    except Exception as e:
        logger.error("Error generating LiveKit token: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate token: {e}")

# --- Health Check Endpoint ---
@app.get("/")
async def root():
    get_engine()  # Raises 503 if engine isn't ready
    return {"status": "ok", "engine_ready": _engine_ready}

if __name__ == "__main__":
    uvicorn.run("real_time_webgui:app", host="0.0.0.0", port=3000, reload=True)
