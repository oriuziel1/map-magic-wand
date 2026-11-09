"""XYZ tile fetching (with disk cache), Web-Mercator math and image mosaics."""
from __future__ import annotations

import asyncio
import io
import math
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from .config import CONFIG
from .net import SSL_CONTEXT

TILE = 256
MAX_LAT = 85.05112878


# ---------------------------------------------------------------- mercator math
def lnglat_to_px(lng: float, lat: float, z: int) -> tuple[float, float]:
    """Lng/lat -> global pixel coords at zoom z (256px tiles)."""
    n = TILE * (2 ** z)
    lat = max(-MAX_LAT, min(MAX_LAT, lat))
    s = math.sin(math.radians(lat))
    x = (lng + 180.0) / 360.0 * n
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def px_to_lnglat_arr(xy: np.ndarray, z: int) -> np.ndarray:
    """Vectorised global pixel -> lng/lat. xy: (N, 2)."""
    n = TILE * (2 ** z)
    lng = xy[:, 0] / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * xy[:, 1] / n))))
    return np.column_stack([lng, lat])


def lnglat_to_px_arr(ll: np.ndarray, z: int) -> np.ndarray:
    n = TILE * (2 ** z)
    lat = np.clip(ll[:, 1], -MAX_LAT, MAX_LAT)
    s = np.sin(np.radians(lat))
    x = (ll[:, 0] + 180.0) / 360.0 * n
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return np.column_stack([x, y])


def meters_per_px(lat: float, z: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


# ---------------------------------------------------------------- tile store
class TileStore:
    def __init__(self) -> None:
        self.layers = CONFIG["layers"]
        self.cache_dir = CONFIG["cache_dir"] / "tiles"
        self.client = httpx.AsyncClient(
            headers={"User-Agent": CONFIG["user_agent"]}, timeout=20, follow_redirects=True, verify=SSL_CONTEXT
        )
        self.sem = asyncio.Semaphore(8)

    async def close(self) -> None:
        await self.client.aclose()

    def max_zoom(self, layer: str) -> int:
        return int(self.layers[layer].get("max_zoom", 19))

    async def get(self, layer: str, z: int, x: int, y: int) -> bytes:
        path = self.cache_dir / layer / str(z) / str(x) / f"{y}.img"
        if path.exists():
            return path.read_bytes()
        url = self.layers[layer]["url"].format(z=z, x=x, y=y)
        async with self.sem:
            r = await self.client.get(url)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return r.content

    async def mosaic(self, layer: str, z: int, x0: int, y0: int, w: int, h: int) -> "Mosaic":
        """Build an RGB image covering global pixel window [x0, x0+w) x [y0, y0+h)."""
        tx0, ty0 = x0 // TILE, y0 // TILE
        tx1, ty1 = (x0 + w - 1) // TILE, (y0 + h - 1) // TILE
        n = 2 ** z
        coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

        async def fetch(tx: int, ty: int):
            if ty < 0 or ty >= n:
                return tx, ty, None
            try:
                data = await self.get(layer, z, tx % n, ty)
                return tx, ty, Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return tx, ty, None

        tiles = await asyncio.gather(*(fetch(tx, ty) for tx, ty in coords))
        canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE), (0, 0, 0))
        for tx, ty, im in tiles:
            if im is not None:
                canvas.paste(im, ((tx - tx0) * TILE, (ty - ty0) * TILE))
        ox, oy = x0 - tx0 * TILE, y0 - ty0 * TILE
        img = np.asarray(canvas.crop((ox, oy, ox + w, oy + h)))
        return Mosaic(layer=layer, z=z, x0=x0, y0=y0, image=img)


@dataclass
class Mosaic:
    layer: str
    z: int
    x0: int  # global pixel origin
    y0: int
    image: np.ndarray  # H x W x 3 uint8

    @property
    def key(self) -> tuple:
        h, w = self.image.shape[:2]
        return (self.layer, self.z, self.x0, self.y0, w, h)

    def to_local(self, lng: float, lat: float) -> tuple[float, float]:
        x, y = lnglat_to_px(lng, lat, self.z)
        return x - self.x0, y - self.y0

    def local_to_lnglat(self, xy: np.ndarray) -> np.ndarray:
        g = np.asarray(xy, dtype=float) + np.array([self.x0, self.y0])
        return px_to_lnglat_arr(g, self.z)

    def lnglat_to_local(self, ll: np.ndarray) -> np.ndarray:
        return lnglat_to_px_arr(np.asarray(ll, dtype=float), self.z) - np.array([self.x0, self.y0])


def window_around(lng: float, lat: float, z: int, size: int) -> tuple[int, int, int, int]:
    """Tile-aligned window of `size` px around a point (aligned so nearby clicks reuse
    the same window -> SAM2 embedding cache hits)."""
    x, y = lnglat_to_px(lng, lat, z)
    tiles = size // TILE
    tx0 = int(x // TILE) - tiles // 2
    ty0 = int(y // TILE) - tiles // 2
    return tx0 * TILE, ty0 * TILE, size, size


def window_for_bounds(minx: float, miny: float, maxx: float, maxy: float, z_max: int,
                      max_px: int, pad_px: int = 32) -> tuple[int, int, int, int, int]:
    """Pick the highest zoom <= z_max at which the lng/lat bounds fit in max_px, return
    (z, x0, y0, w, h)."""
    for z in range(z_max, 0, -1):
        x0, y1 = lnglat_to_px(minx, miny, z)
        x1, y0 = lnglat_to_px(maxx, maxy, z)
        w, h = x1 - x0 + 2 * pad_px, y1 - y0 + 2 * pad_px
        if w <= max_px and h <= max_px:
            return z, int(x0 - pad_px), int(y0 - pad_px), int(math.ceil(w)), int(math.ceil(h))
    raise ValueError("bounds too large")
