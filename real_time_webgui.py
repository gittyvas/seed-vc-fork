import os
import sys
import json
import logging
import time
import asyncio
from pathlib import Path  # NEW: Using pathlib for robust path handling
import numpy as np
import torch
import uvicorn
import sounddevice as sd
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from agora_token_builder import RtcTokenBuilder

# --- Configuration ---
AGORA_APP_ID = os.getenv("AGORA_APP_ID", "33e50001273c408fb2f3408415506d75")
AGORA_APP_CERT = os.getenv("AGORA_APP_CERT", "99c1b87b9daa40e1bb4a1dc5c84f503a")
CONFIG_PATH = os.path.join(os.getcwd(), "configs", "v2", "vc_wrapper.yaml")
GPU = 0
FP16 = True
LOG_LEVEL = "INFO"
MODEL_SAMPLING_RATE = 16000

# NEW: Define path to reference audio directory robustly
# This assumes your 'examples' folder is in the same directory as this script.
APP_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = APP_DIR / "examples" / "reference"

# --- App Setup ---
app = FastAPI(title="Seed-VC v2 Realtime Voice Changer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
# NEW: Mount the reference audio directory so the frontend can fetch files from it.
if REFERENCE_DIR.is_dir():
    app.mount("/references", StaticFiles(directory=REFERENCE_DIR), name="references")
else:
    logging.warning(f"Reference audio directory not found at: {REFERENCE_DIR}")

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
_reference_audio = None

@app.on_event("startup")
async def startup_event():
    global _engine_ready, _model_set
    logger.info("🔄 Initializing voice conversion engine...")
    class ArgsObj:
        fp16 = FP16; checkpoint_path = None; config_path = CONFIG_PATH; gpu = GPU
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

# --- Routes ---
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# NEW: API endpoint to list available reference .wav files
@app.get("/api/references")
async def get_reference_files():
    if not REFERENCE_DIR.is_dir():
        return JSONResponse(status_code=404, content={"error": "Reference directory not found on server."})
    try:
        wav_files = [f.name for f in REFERENCE_DIR.glob("*.wav")]
        return wav_files
    except Exception as e:
        logger.error(f"Error reading reference directory: {e}")
        return JSONResponse(status_code=500, content={"error": "Could not read reference files."})

@app.get("/api/devices")
async def list_devices():
    # This endpoint remains the same
    try:
        devices = sd.query_devices()
        input_devices = [{"name": d["name"], "id": i} for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        output_devices = [{"name": d["name"], "id": i} for i, d in enumerate(devices) if d["max_output_channels"] > 0]
        return {"input_devices": input_devices, "output_devices": output_devices}
    except Exception as e:
        logger.error(f"Could not query audio devices: {e}")
        return JSONResponse(status_code=500, content={"message": "Could not query audio devices"})

@app.get("/api/model_info")
async def get_model_info():
    return {"sampling_rate": MODEL_SAMPLING_RATE}

@app.post("/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    global _reference_audio
    try:
        data = await file.read()
        import soundfile as sf
        import io
        audio_np, sr = sf.read(io.BytesIO(data), dtype="float32")
        if sr != MODEL_SAMPLING_RATE:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=MODEL_SAMPLING_RATE)
        _reference_audio = audio_np
        logger.info(f"🎤 Cached reference audio '{file.filename}' ({len(audio_np)} samples)")
        return {"status": "success", "message": f"Reference '{file.filename}' loaded."}
    except Exception as e:
        logger.error("Error loading reference clip: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid reference audio file: {e}")

@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    # This endpoint logic remains the same
    global _reference_audio
    await ws.accept()
    logger.info("🔗 WebSocket client connected")
    settings = {}
    try:
        initial_config = await ws.receive_json()
        if initial_config.get("type") == "config":
            settings = initial_config.get("data", {})
            logger.info(f"Received initial GUI settings: {settings}")
        else:
            await ws.close(code=1008, reason="First message must be a config object.")
            return
        while True:
            message = await ws.receive()
            if "text" in message:
                update_data = json.loads(message["text"])
                if update_data.get("type") == "config":
                    settings.update(update_data.get("data", {}))
                    logger.info(f"Updated settings: {settings}")
                continue
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()
                if _reference_audio is None:
                    await ws.send_json({"error": "No reference loaded yet", "type": "error"})
                    continue
                try:
                    mic_gain = settings.get("mic_input_gain", 1.0)
                    audio_np *= mic_gain
                    start = time.perf_counter()
                    out_audio = custom_infer(
                        model_set=get_engine(), reference_wav=_reference_audio,
                        new_reference_wav_name="session_ref",  input_wav_res=audio_np,
                        block_frame_16k=len(audio_np) // 320,
                        return_length=len(audio_np) // 320, skip_head=0, skip_tail=0,
                        diffusion_steps=settings.get("diffusion_steps", 20),
                        inference_cfg_rate=settings.get("inference_cfg_rate", 1.0),
                        max_prompt_length=settings.get("max_prompt_length", 3)
                    )
                
                    infer_ms = (time.perf_counter() - start) * 1000
                    await ws.send_json({
                        "type": "metadata",
                        "inference_time": infer_ms,
                        "algorithm_delay": (len(audio_np) / MODEL_SAMPLING_RATE) * 1000
                    })
                    await ws.send_bytes(out_audio.astype(np.float32).tobytes())

                except Exception as e:
                    logger.error("❌ Inference error: %s", e, exc_info=True)
                    await ws.send_json({
                        "error": "processing_failed",
                        "details": str(e),
                        "type": "error"
                    })
                    continue  # keep socket alive for next messages

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")

    except Exception as e:
        logger.error("💥 Unexpected WebSocket error: %s", e, exc_info=True)


if __name__ == "__main__":
    uvicorn.run("real_time_webgui:app", host="0.0.0.0", port=3000, reload=True)
