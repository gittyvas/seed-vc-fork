import os
import sys
import json
import logging
import time
import asyncio
import numpy as np
import torch
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from agora_token_builder import RtcTokenBuilder

# --- Configuration ---
AGORA_APP_ID = "33e50001273c408fb2f3408415506d75"
AGORA_APP_CERT = "99c1b87b9daa40e1bb4a1dc5c84f503a"
CONFIG_PATH = os.path.join(os.getcwd(), "configs", "v2", "vc_wrapper.yaml")
GPU = 0
FP16 = True
LOG_LEVEL = "INFO"

# --- App Setup ---
app = FastAPI(title="Seed-VC v2 Realtime Voice Changer")
app.mount("/static", StaticFiles(directory="static"), name="static")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("seedvc.app")

# --- Import Engine ---
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from vc_engine import init_engine, custom_infer
except ImportError as e:
    raise RuntimeError(f"Failed to import vc_engine: {e}")

# --- Model Engine ---
_engine_ready = False
_model_set = None
_reference_audio = None  # global cache for reference voice

@app.on_event("startup")
async def startup_event():
    global _engine_ready, _model_set
    logger.info("🔄 Initializing voice conversion engine...")
    class ArgsObj:
        fp16 = FP16
        checkpoint_path = None
        config_path = CONFIG_PATH
        gpu = GPU
    try:
        _model_set = init_engine(ArgsObj())
        _engine_ready = True
        logger.info("✅ Model engine initialized successfully.")
    except Exception as e:
        logger.error("❌ Failed to initialize model engine: %s", e, exc_info=True)

def get_engine():
    if not _engine_ready:
        raise HTTPException(status_code=503, detail="Model engine is not ready.")
    return _model_set

# --- Root Route ---
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# --- Agora Token Endpoint ---
@app.get("/get_token")
async def get_agora_token(
    uid: int = Query(0, description="Numeric user ID"),
    channel: str = Query("test_channel", description="Agora channel name")
):
    try:
        expire_time = int(time.time()) + 3600
        token = RtcTokenBuilder.buildTokenWithUid(
            AGORA_APP_ID,
            AGORA_APP_CERT,
            channel,
            uid,
            1,  # Role_Attendee
            expire_time
        )
        return {"token": token, "app_id": AGORA_APP_ID, "channel": channel, "uid": uid}
    except Exception as e:
        logger.error("Failed to generate Agora token: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate token")

# --- Upload Reference Voice ---
@app.post("/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    """
    Upload a reference clip for zero-shot voice cloning.
    """
    global _reference_audio
    try:
        data = await file.read()
        import soundfile as sf
        import io
        audio_np, sr = sf.read(io.BytesIO(data), dtype="float32")
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        _reference_audio = audio_np
        logger.info(f"🎤 Cached reference audio ({len(audio_np)} samples @ 16kHz)")
        return {"status": "ok", "samples": len(audio_np)}
    except Exception as e:
        logger.error("Error loading reference clip: %s", e)
        raise HTTPException(status_code=400, detail="Invalid reference audio")

# --- WebSocket for Live Streaming ---
@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    global _reference_audio
    await ws.accept()
    logger.info("🔗 WebSocket client connected for audio stream")

    try:
        while True:
            data = await ws.receive_bytes()
            audio_np = np.frombuffer(data, dtype=np.float32)

            if _reference_audio is None:
                # cannot process without reference
                await ws.send_json({"error": "No reference uploaded yet"})
                continue

            try:
                # --- Start timing the inference process ---
                start = time.perf_counter()

                # Process audio chunk
                out_audio = custom_infer(
                    model_set=get_engine(),
                    reference_wav=_reference_audio,
                    new_reference_wav_name="session_ref",
                    input_wav=audio_np,
                    return_length=len(audio_np) // 320,
                    skip_head=0,
                    skip_tail=0,
                    diffusion_steps=20,  # lower for faster response
                    inference_cfg_rate=1.0,
                    max_prompt_length=3
                )
                
                # --- Calculate inference time in milliseconds ---
                infer_ms = (time.perf_counter() - start) * 1000

                # --- Send timing metadata first ---
                await ws.send_json({
                    "meta": {
                        "algorithm_delay": len(audio_np) / 16000 * 1000,
                        "inference_time": infer_ms
                    }
                })
                
                # --- Then send the processed audio bytes ---
                await ws.send_bytes(out_audio.astype(np.float32).tobytes())

            except Exception as e:
                logger.error("Inference error: %s", e)
                await ws.send_json({"error": "processing_failed"})

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e)

if __name__ == "__main__":
    uvicorn.run("real_time_webgui:app", host="0.0.0.0", port=3000, reload=True)
