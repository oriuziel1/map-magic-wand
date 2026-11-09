from . import net  # noqa: F401  (must be first: OS cert store for all HTTPS)

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import CONFIG
from .sam import ENGINE
from .tiles import TileStore
from .wand import WandRequest, run_wand

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["tiles"] = TileStore()
    threading.Thread(target=ENGINE.load, daemon=True).start()  # load SAM2 without blocking startup
    yield
    await STATE["tiles"].close()


app = FastAPI(title="Map Magic Wand POC", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/config")
def get_config():
    return {
        "center": CONFIG["map"]["center"],
        "zoom": CONFIG["map"]["zoom"],
        "layers": {
            k: {"name": v["name"], "max_zoom": v.get("max_zoom", 19), "attribution": v.get("attribution", ""),
                "tiles": f"/api/tiles/{k}/{{z}}/{{x}}/{{y}}"}
            for k, v in CONFIG["layers"].items()
        },
    }


@app.get("/api/health")
def health():
    return {"sam_ready": ENGINE.ready, "sam_loading": ENGINE.loading, "sam_error": ENGINE.error,
            "device": ENGINE.device, "model": CONFIG["sam"]["model"]}


@app.get("/api/tiles/{layer}/{z}/{x}/{y}")
async def tile(layer: str, z: int, x: int, y: int):
    if layer not in CONFIG["layers"]:
        raise HTTPException(404, "unknown layer")
    try:
        data = await STATE["tiles"].get(layer, z, x, y)
    except Exception as ex:
        raise HTTPException(502, f"tile fetch failed: {ex}")
    mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return Response(data, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/wand")
async def wand(req: WandRequest):
    return await run_wand(req, STATE["tiles"])
