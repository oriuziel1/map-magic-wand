"""Magic-wand strategy: OSM vectors first, SAM2 on imagery as fallback / refinement."""
from __future__ import annotations

import asyncio
import time
from typing import Literal

import numpy as np
import shapely
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from shapely.geometry import Point, box
from shapely.ops import unary_union

from . import osm
from .config import CONFIG
from .geometry import (area_m2, clean, clip_to, from_geojson, mask_to_polygons, pick_polygon,
                       reproject, to_feature, utm_transformers)
from .sam import ENGINE
from .tiles import TileStore, meters_per_px, window_around, window_for_bounds


class WandRequest(BaseModel):
    lng: float
    lat: float
    layer: Literal["ortho", "street"] = "ortho"
    zoom: float = 17  # MapLibre zoom (512px world) -> tile zoom = zoom + 1
    view_bbox: list[float] | None = None  # [w, s, e, n]
    clip: dict | None = None  # GeoJSON Polygon / Feature
    mode: Literal["auto", "osm", "ai"] = "auto"
    points: list[list[float]] = []  # extra refine prompts [lng, lat, label(1|0)]
    simplify: float | None = None  # px tolerance


async def run_wand(req: WandRequest, store: TileStore) -> dict:
    t0 = time.time()
    msgs: list[str] = []
    clip = from_geojson(req.clip)
    click = Point(req.lng, req.lat)
    if clip is not None and not clip.buffer(1e-6).contains(click):
        msgs.append("Click is outside the clip polygon - clip ignored.")
        clip = None
    view = box(*req.view_bbox) if req.view_bbox else click.buffer(0.003)
    tile_z = int(min(round(req.zoom) + 1, store.max_zoom(req.layer)))
    simplify = req.simplify if req.simplify is not None else CONFIG["wand"]["simplify_px"]

    result: dict | None = None
    hit = None
    if req.mode != "ai" and not req.points:
        bound = clip if clip is not None else view
        b = bound.bounds
        if b[2] - b[0] > 0.03 or b[3] - b[1] > 0.03:  # zoomed far out: limit road search
            bound = bound.intersection(click.buffer(0.01))
        try:
            areas, roads = await osm.load_features(req.lng, req.lat, bound.bounds)
            hit = osm.find_hit(req.lng, req.lat, areas, roads, bound, clip, allow_large=req.mode == "osm")
        except osm.OsmUnavailable as ex:
            msgs.append(f"{ex} - falling back to SAM2.")

    if hit is not None:
        osm_feat = to_feature(hit.geom, source="osm", kind=hit.kind, osm_tags=hit.tags,
                              osm_ids=hit.osm_ids)
        result = {"feature": osm_feat, "alternatives": [osm_feat], "selected": 0}
        if hit.kind == "road" and req.layer == "ortho" and req.mode == "auto":
            if ENGINE.ready:
                try:
                    refined = await _sam_road(req, hit, clip, store, tile_z, simplify)
                    if refined is not None:
                        result = {"feature": refined, "alternatives": [refined, osm_feat], "selected": 0}
                    else:
                        msgs.append("SAM2 road refinement looked unreliable - using OSM width buffer.")
                except Exception as ex:
                    msgs.append(f"SAM2 refinement failed ({ex}) - using OSM width buffer.")
            else:
                msgs.append("SAM2 not ready - using OSM width buffer.")
    elif req.mode == "osm":
        msgs.append("No OSM feature found here.")
    else:
        if not ENGINE.ready:
            msgs.append(ENGINE.error or "SAM2 model is still loading, try again in a moment.")
        else:
            try:
                result = await _sam_click(req, clip, store, tile_z, simplify)
            except RuntimeError as ex:
                msgs.append(str(ex))
                result = None
            if result is None and not msgs:
                msgs.append("SAM2 found nothing at this point.")

    out = result or {"feature": None, "alternatives": [], "selected": 0}
    out["messages"] = msgs
    out["debug"] = {"tile_zoom": tile_z, "ms": int((time.time() - t0) * 1000), "device": ENGINE.device}
    return out


# ---------------------------------------------------------------- SAM2 paths
SAM_TIMEOUT_S = 20


async def _predict(mosaic, pts, multimask):
    try:
        return await asyncio.wait_for(
            run_in_threadpool(ENGINE.predict, mosaic.key, mosaic.image, pts, None, multimask), SAM_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise RuntimeError(f"SAM2 timed out after {SAM_TIMEOUT_S}s")
async def _mosaic_for(req: WandRequest, clip, store: TileStore, tile_z: int):
    wcfg = CONFIG["wand"]
    if clip is not None:
        z, x0, y0, w, h = window_for_bounds(*clip.bounds, z_max=tile_z, max_px=wcfg["max_mosaic_px"])
    else:
        z = tile_z
        x0, y0, w, h = window_around(req.lng, req.lat, z, wcfg["mosaic_px"])
    return await store.mosaic(req.layer, z, x0, y0, w, h)


def _vectorise(mosaic, mask, click_xy, simplify, clip, keep_all=False, close_px=3):
    polys = mask_to_polygons(mask, close_px=close_px)
    if keep_all:
        geom = unary_union(polys) if polys else None
    else:
        geom = pick_polygon(polys, click_xy)
    geom = clean(geom, simplify)
    if geom is None:
        return None
    geom = shapely.transform(geom, mosaic.local_to_lnglat)
    return clip_to(geom, clip)


async def _sam_click(req: WandRequest, clip, store: TileStore, tile_z: int, simplify: float):
    mosaic = await _mosaic_for(req, clip, store, tile_z)
    h, w = mosaic.image.shape[:2]
    cx, cy = mosaic.to_local(req.lng, req.lat)
    pts = [(cx, cy, 1)]
    for lng, lat, label in req.points:
        x, y = mosaic.to_local(lng, lat)
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y, int(label)))
    results = await _predict(mosaic, pts, len(pts) == 1)
    feats = []
    for i, (mask, score) in enumerate(results):
        g = _vectorise(mosaic, mask, (cx, cy), simplify, clip)
        if g is not None:
            feats.append(to_feature(g, source="sam2", score=round(score, 3), rank=i))
    if not feats:
        return None
    best = feats[0]
    feats.sort(key=lambda f: f["properties"]["area_m2"])  # small -> large = shrink/grow
    return {"feature": best, "alternatives": feats, "selected": feats.index(best)}


async def _sam_road(req: WandRequest, hit, clip, store: TileStore, tile_z: int, simplify: float):
    """Use the OSM centerline as SAM2 point prompts so the polygon follows the real pavement,
    then keep it inside a corridor around the centerline (stops leaks into parking lots)."""
    wcfg = CONFIG["wand"]
    z, x0, y0, w, h = window_for_bounds(*hit.geom.bounds, z_max=tile_z, max_px=wcfg["max_mosaic_px"])
    mosaic = await store.mosaic(req.layer, z, x0, y0, w, h)
    line_px = shapely.transform(hit.centerline, mosaic.lnglat_to_local)
    parts = list(getattr(line_px, "geoms", [line_px]))
    parts = [p for p in parts if p.geom_type == "LineString" and p.length > 0]
    total = sum(p.length for p in parts)
    if total == 0:
        return None
    n = int(np.clip(total / 120, 3, 12))
    pts = []
    for p in parts:
        k = max(1, round(n * p.length / total))
        for d in np.linspace(0.08, 0.92, k):
            q = p.interpolate(d, normalized=True)
            if 0 <= q.x < w and 0 <= q.y < h:
                pts.append((q.x, q.y, 1))
    cx, cy = mosaic.to_local(req.lng, req.lat)
    pts.append((cx, cy, 1))
    results = await _predict(mosaic, pts, False)
    mask, score = results[0]
    close_px = int(hit.width_m * 0.8 / meters_per_px(req.lat, z))
    g = _vectorise(mosaic, mask, (cx, cy), simplify, None, keep_all=True, close_px=close_px)
    if g is None:
        return None
    fwd, inv = utm_transformers(req.lng, req.lat)
    corridor = reproject(reproject(hit.centerline, fwd).buffer(hit.width_m * 0.9), inv)
    bound = clip if clip is not None else hit.geom.envelope.buffer(0)
    g = clip_to(clip_to(g, corridor), bound)
    if g is None:
        return None
    ratio = area_m2(g) / max(area_m2(hit.geom), 1)
    if ratio < 0.35:
        return None
    return to_feature(g, source="osm+sam2", kind="road", osm_tags=hit.tags, osm_ids=hit.osm_ids,
                      score=round(score, 3))
