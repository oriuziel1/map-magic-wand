"""Mask -> polygon vectorisation and geo helpers."""
from __future__ import annotations

import cv2
import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry


# ---------------------------------------------------------------- metric projection
def utm_transformers(lng: float, lat: float) -> tuple[Transformer, Transformer]:
    zone = int((lng + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    return fwd, inv


def reproject(geom: BaseGeometry, tr: Transformer) -> BaseGeometry:
    return shapely.transform(geom, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))


def area_m2(geom: BaseGeometry) -> float:
    if geom.is_empty:
        return 0.0
    c = geom.centroid
    fwd, _ = utm_transformers(c.x, c.y)
    return float(reproject(geom, fwd).area)


# ---------------------------------------------------------------- mask -> polygons
def mask_to_polygons(mask: np.ndarray, min_area_px: float = 30, close_px: int = 3) -> list[Polygon]:
    """Binary mask -> list of polygons (pixel coords) with holes. close_px bridges gaps
    (e.g. cars / shadows breaking up a road)."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, close_px), max(3, close_px)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    polys: list[Polygon] = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or len(c) < 3:
            continue  # holes handled with their parent
        shell = c[:, 0, :].astype(float)
        holes = []
        child = hier[i][2]
        while child != -1:
            hc = contours[child]
            if len(hc) >= 3 and cv2.contourArea(hc) >= min_area_px:
                holes.append(hc[:, 0, :].astype(float))
            child = hier[child][0]
        p = Polygon(shell, holes)
        if not p.is_valid:
            p = shapely.make_valid(p)
        if p.area >= min_area_px:
            polys.extend(_polys_only(p))
    return polys


def _polys_only(g: BaseGeometry) -> list[Polygon]:
    if isinstance(g, Polygon):
        return [g] if not g.is_empty else []
    if hasattr(g, "geoms"):
        out = []
        for gg in g.geoms:
            out.extend(_polys_only(gg))
        return out
    return []


def pick_polygon(polys: list[Polygon], click_xy: tuple[float, float] | None) -> BaseGeometry | None:
    """Polygon containing the click, else the largest."""
    if not polys:
        return None
    if click_xy is not None:
        pt = Point(click_xy)
        inside = [p for p in polys if p.buffer(1).contains(pt)]
        if inside:
            return max(inside, key=lambda p: p.area)
    return max(polys, key=lambda p: p.area)


def clean(geom: BaseGeometry | None, simplify: float) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    if simplify > 0:
        geom = geom.simplify(simplify, preserve_topology=True)
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    polys = _polys_only(geom)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def local_px_to_lnglat(geom: BaseGeometry, to_lnglat) -> BaseGeometry:
    return shapely.transform(geom, to_lnglat)


def clip_to(geom: BaseGeometry | None, clip: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None or clip is None:
        return geom
    out = geom.intersection(clip)
    polys = _polys_only(out)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def to_feature(geom: BaseGeometry, **props) -> dict:
    props.setdefault("area_m2", round(area_m2(geom), 1))
    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def from_geojson(g: dict | None) -> BaseGeometry | None:
    if not g:
        return None
    if g.get("type") == "Feature":
        g = g["geometry"]
    geom = shape(g)
    return geom if geom.is_valid else shapely.make_valid(geom)
