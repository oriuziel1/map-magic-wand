"""OpenStreetMap vector lookup - the "it already exists on the map" path.

Two interchangeable providers (config `osm.provider`):
  * vector_tiles - OpenMapTiles-schema MVT tiles (OpenFreeMap online, or any local tile server /
                   mbtiles for offline). Fast and reliable.
  * overpass     - Overpass API (richer tags/names, but public servers are often overloaded).
Both return the same normalised (areas, roads) lists.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, polygonize, unary_union

from .config import CONFIG
from .geometry import _polys_only, reproject, utm_transformers
from .net import SSL_CONTEXT
from .tiles import TILE, lnglat_to_px, px_to_lnglat_arr

AREA_KEYS = ("building", "landuse", "leisure", "natural", "amenity", "water", "man_made", "aeroway")
ROAD_WIDTH = {
    "motorway": 14, "trunk": 12, "primary": 10, "secondary": 9, "tertiary": 8,
    "motorway_link": 6, "trunk_link": 6, "primary_link": 6, "secondary_link": 6, "tertiary_link": 6,
    "residential": 7, "unclassified": 6, "living_street": 5, "service": 4, "pedestrian": 4,
    "track": 3, "footway": 2, "path": 2, "cycleway": 2, "steps": 2,
}
# OpenMapTiles transportation class -> OSM highway value
OMT_CLASS = {"minor": "residential", "busway": "service", "bus_guideway": "service", "raceway": "service"}
MINOR_WAYS = {"footway", "path", "cycleway", "steps", "pedestrian", "track", "bridleway"}
NON_ROAD = {"rail", "transit", "ferry", "aerialway", "pier"}
SMALL_AREA_M2 = 20_000
IGNORED_AREA_CLASSES = {"suburb", "neighbourhood", "quarter", "city_block", "residential"}

Areas = list[tuple[BaseGeometry, dict, int]]
Roads = list[tuple[LineString, dict, int]]


class OsmUnavailable(Exception):
    pass


@dataclass
class OsmHit:
    kind: str  # "area" | "road"
    geom: BaseGeometry  # lng/lat polygon
    tags: dict
    centerline: BaseGeometry | None = None  # roads only (lng/lat)
    width_m: float = 0.0
    osm_ids: list = field(default_factory=list)


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"User-Agent": CONFIG["user_agent"]}, timeout=timeout,
                             verify=SSL_CONTEXT, follow_redirects=True)


async def load_features(lng: float, lat: float, bbox: tuple[float, float, float, float]) -> tuple[Areas, Roads]:
    provider = CONFIG["osm"].get("provider", "vector_tiles")
    if provider == "overpass":
        return parse_overpass(await fetch_overpass(lng, lat, bbox))
    return await fetch_vector_tiles(lng, lat, bbox)


# ================================================================ vector tiles
_tile_url_cache: dict[str, str] = {}


async def _vt_template(client: httpx.AsyncClient) -> str:
    url = CONFIG["osm"]["vector_tiles"]["url"]
    if "{z}" in url:
        return url
    if url not in _tile_url_cache:  # TileJSON endpoint
        try:
            r = await client.get(url)
            r.raise_for_status()
            _tile_url_cache[url] = r.json()["tiles"][0]
        except Exception as ex:
            raise OsmUnavailable(f"Vector TileJSON unavailable: {ex}") from ex
    return _tile_url_cache[url]


async def fetch_vector_tiles(lng: float, lat: float, bbox) -> tuple[Areas, Roads]:
    import mapbox_vector_tile as mvt

    z = int(CONFIG["osm"]["vector_tiles"].get("zoom", 14))
    w, s, e, n = bbox
    x0, y0 = (int(v // TILE) for v in lnglat_to_px(w, n, z))
    x1, y1 = (int(v // TILE) for v in lnglat_to_px(e, s, z))
    if (x1 - x0 + 1) * (y1 - y0 + 1) > 9:  # too big: only the 3x3 around the click
        cx, cy = (int(v // TILE) for v in lnglat_to_px(lng, lat, z))
        x0, x1, y0, y1 = cx - 1, cx + 1, cy - 1, cy + 1
    cache_root = CONFIG["cache_dir"] / "vector"

    async with _client(20) as client:
        template = await _vt_template(client)

        async def get(tx, ty):
            p = cache_root / str(z) / str(tx) / f"{ty}.pbf"
            if p.exists():
                return tx, ty, p.read_bytes()
            try:
                r = await client.get(template.format(z=z, x=tx, y=ty))
                r.raise_for_status()
            except Exception as ex:
                raise OsmUnavailable(f"Vector tile fetch failed: {ex}") from ex
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content)
            return tx, ty, r.content

        tiles = await asyncio.gather(*(get(tx, ty) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)))

    pieces: dict[tuple, list] = defaultdict(list)  # (layer, id, kind) -> [(tile, geom)]
    tagmap: dict[tuple, dict] = {}
    near = Point(lng, lat).buffer(CONFIG["osm"]["building_search_m"] / 111_000)
    bbox_poly = box(*bbox)
    for tx, ty, data in tiles:
        key = (z, tx, ty)
        decoded = _decoded.get(key)
        if decoded is None:
            decoded = mvt.decode(data, default_options={"y_coord_down": True})
            _decoded[key] = decoded
            if len(_decoded) > 64:
                _decoded.pop(next(iter(_decoded)))
        for layer, lyr in decoded.items():
            if layer not in ("building", "transportation", "landuse", "landcover", "park", "water", "aeroway"):
                continue
            ext = lyr.get("extent", 4096)

            def to_ll(c, tx=tx, ty=ty, ext=ext):
                g = np.column_stack([tx * TILE + c[:, 0] / ext * TILE, ty * TILE + c[:, 1] / ext * TILE])
                return px_to_lnglat_arr(g, z)

            for fi, f in enumerate(lyr["features"]):
                gt = f["geometry"]["type"]
                props = f.get("properties", {})
                if gt in ("Point", "MultiPoint"):
                    continue
                kind = "line" if "LineString" in gt else "area"
                try:
                    geom = shapely.transform(shape(f["geometry"]), to_ll)
                except Exception:
                    continue
                if kind == "area":  # only area parts at/near the click matter
                    geom = _parts_near(geom, near)
                    if geom is None:
                        continue
                elif not geom.intersects(bbox_poly):
                    continue
                if layer == "transportation":
                    cls = props.get("class", "")
                    if cls in NON_ROAD or cls.endswith("_construction"):
                        continue
                    hw = OMT_CLASS.get(cls, cls)
                    if props.get("ramp") == 1 and hw in ("motorway", "trunk", "primary", "secondary", "tertiary"):
                        hw += "_link"
                    tags = {"highway": hw, **{k: v for k, v in props.items() if k in ("brunnel", "oneway", "surface")}}
                    if kind == "area":
                        tags["area"] = "yes"
                elif layer == "building":
                    tags = {"building": "yes"}
                else:
                    key = {"landcover": "natural", "park": "leisure"}.get(layer, layer)
                    tags = {key: props.get("class", "yes"), **({"subclass": props["subclass"]} if "subclass" in props else {})}
                # features with an OSM-derived id are stitched across tiles; others stay unique
                k = (layer, f.get("id") or f"{tx}/{ty}/{fi}", kind)
                pieces[k].append(((tx, ty), geom))
                tagmap[k] = tags

    areas: Areas = []
    roads: Roads = []
    for (layer, fid, kind), geoms in pieces.items():
        tags = tagmap[(layer, fid, kind)]
        if kind == "area":
            for poly in _stitch_areas(geoms):
                areas.append((poly, tags, fid))
        else:
            g = shapely.line_merge(unary_union([g for _, g in geoms]))
            for part in getattr(g, "geoms", [g]):
                if part.geom_type == "LineString" and part.length > 0:
                    roads.append((part, tags, fid))
    return areas, roads


_decoded: dict = {}


def _parts_near(g: BaseGeometry, near: BaseGeometry) -> BaseGeometry | None:
    parts = [p for p in getattr(g, "geoms", [g]) if p.intersects(near)]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(
        [q for p in parts for q in _polys_only(p)])


def _stitch_areas(geoms: list[tuple[tuple, BaseGeometry]]) -> list[BaseGeometry]:
    """Vector tiles may pack many buildings into one MultiPolygon feature, and cut polygons at
    tile borders (with an overlapping buffer). Explode into parts, then merge only parts from
    *different* tiles that genuinely overlap - never neighbours that merely touch."""
    parts = []
    for tile, g in geoms:
        for p in _polys_only(shapely.make_valid(g)):
            parts.append((tile, p))
    if len(geoms) == 1 or len({t for t, _ in parts}) == 1:
        return [p for _, p in parts]
    parent = list(range(len(parts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree([p for _, p in parts])
    for i, (ti, p) in enumerate(parts):
        for j in tree.query(p):
            j = int(j)
            if j > i and parts[j][0] != ti and p.intersection(parts[j][1]).area > 1e-12:
                parent[find(j)] = find(i)
    groups: dict[int, list] = defaultdict(list)
    for i, (_, p) in enumerate(parts):
        groups[find(i)].append(p)
    out = []
    for ps in groups.values():
        out.extend(_polys_only(unary_union(ps)) if len(ps) > 1 else ps)
    return out


# ================================================================ overpass
async def fetch_overpass(lng: float, lat: float, bbox) -> dict:
    cfg = CONFIG["osm"]
    w, s, e, n = bbox
    r = cfg["building_search_m"]
    keys = "|".join(AREA_KEYS)
    q = f"""[out:json][timeout:{cfg['timeout_s']}];
(
  way["building"](around:{r},{lat},{lng});
  relation["building"](around:{r},{lat},{lng});
  way["highway"]({s},{w},{n},{e});
)->.near;
is_in({lat},{lng})->.a;
(
  way(pivot.a)[~"^({keys})$"~"."];
  relation(pivot.a)["type"="multipolygon"][~"^({keys})$"~"."];
)->.areas;
(.near; .areas;);
out tags geom;"""
    cache = CONFIG["cache_dir"] / "overpass" / (hashlib.sha1(q.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    errors = []
    async with _client(cfg["timeout_s"] + 5) as client:
        for url in cfg["overpass_urls"]:
            try:
                resp = await client.post(url, data={"data": q})
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as ex:
                errors.append(f"{url.split('/')[2]}: {ex.__class__.__name__}")
        else:
            raise OsmUnavailable("Overpass unavailable (" + "; ".join(errors) + ")")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def _coords(geom_list) -> list[tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in geom_list if p]


def parse_overpass(data: dict) -> tuple[Areas, Roads]:
    areas, roads = [], []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            cs = _coords(el["geometry"])
            if len(cs) < 2:
                continue
            closed = len(cs) >= 4 and cs[0] == cs[-1]
            if "highway" in tags and tags.get("area") != "yes":
                roads.append((LineString(cs), tags, el["id"]))
            elif closed:
                p = shapely.make_valid(Polygon(cs))
                if not p.is_empty:
                    areas.append((p, tags, el["id"]))
        elif el["type"] == "relation":
            outers, inners = [], []
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("geometry"):
                    cs = _coords(m["geometry"])
                    if len(cs) >= 2:
                        (inners if m.get("role") == "inner" else outers).append(LineString(cs))
            if not outers:
                continue
            outer = unary_union(list(polygonize(linemerge(outers))))
            if inners:
                outer = outer.difference(unary_union(list(polygonize(linemerge(inners)))))
            if not outer.is_empty:
                areas.append((outer, tags, el["id"]))
    return areas, roads


# ================================================================ hit test
def road_width(tags: dict) -> float:
    m = re.match(r"^\s*([\d.]+)", str(tags.get("width", "") or ""))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    lanes = re.match(r"^\s*(\d+)", str(tags.get("lanes", "") or ""))
    if lanes:
        return int(lanes.group(1)) * 3.3
    return float(ROAD_WIDTH.get(tags.get("highway", ""), 6))


def find_hit(lng: float, lat: float, areas: Areas, roads: Roads, road_bound: BaseGeometry,
             area_clip: BaseGeometry | None, want: str = "any", allow_large: bool = False) -> OsmHit | None:
    """want: any | road | area. Large areas (parks, landuse) only when allow_large - otherwise the
    caller falls back to SAM2, which outlines what is actually under the cursor."""
    fwd, inv = utm_transformers(lng, lat)
    pt = Point(lng, lat)
    pm = reproject(pt, fwd)

    containing = []
    for poly, tags, oid in areas:
        if any(v in IGNORED_AREA_CLASSES for k, v in tags.items() if k != "building"):
            continue
        if poly.contains(pt):
            containing.append((reproject(poly, fwd).area, poly, tags, oid))
    containing.sort(key=lambda t: t[0])

    roads_m = [(reproject(l, fwd), t, i) for l, t, i in roads]
    best_road = None
    for idx, (lm, tags, oid) in enumerate(roads_m):
        wdt = road_width(tags)
        d = lm.distance(pm) - wdt / 2
        if d > CONFIG["osm"]["road_hit_m"]:
            continue
        score = d + (4 if tags.get("highway") in MINOR_WAYS else 0)  # prefer real roads over sidewalks
        if best_road is None or score < best_road[0]:
            best_road = (score, idx, wdt)

    def area_hit(c):
        _, poly, tags, oid = c
        g = poly.intersection(area_clip) if area_clip is not None else poly
        return OsmHit("area", g, tags, osm_ids=[oid])

    # priority: building > road line > small area > highway-area (plazas) > any large area
    buildings = [c for c in containing if "building" in c[2]]
    small = [c for c in containing if "building" not in c[2] and "highway" not in c[2] and c[0] < SMALL_AREA_M2]
    hw_areas = [c for c in containing if "highway" in c[2]]
    if want == "area":
        return area_hit((buildings or small or hw_areas or containing)[0]) if containing else None
    if buildings:
        return area_hit(buildings[0])
    if best_road is not None:
        return _road_hit(best_road, roads_m, road_bound, fwd, inv)
    if want == "road":
        return None
    for group in (small, hw_areas, containing if allow_large else []):
        if group:
            return area_hit(group[0])
    return None


def _angle(v1, v2) -> float:
    a = math.degrees(math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0]))
    return abs((a + 180) % 360 - 180)


def _follow_stroke(start: int, roads_m, tol: float = 2.0, max_turn: float = 40.0) -> list[int]:
    """Grow from a road segment through junctions, always taking the straightest continuation of
    the same road class (a 'stroke'). Keeps a click on one street from selecting the whole grid."""
    cls = roads_m[start][1].get("highway")
    cands = [i for i, (_, t, _) in enumerate(roads_m) if t.get("highway") == cls]
    chosen, used = [start], {start}
    for at_end in (0, -1):
        cur, end = start, at_end
        for _ in range(100):
            cs = roads_m[cur][0].coords
            p = np.array(cs[end])
            prev = np.array(cs[1] if end == 0 else cs[-2])
            heading = p - prev
            best = None
            for i in cands:
                if i in used:
                    continue
                c = roads_m[i][0].coords
                if np.hypot(*(np.array(c[0]) - p)) < tol:
                    nxt, new_end = np.array(c[1]), -1
                elif np.hypot(*(np.array(c[-1]) - p)) < tol:
                    nxt, new_end = np.array(c[-2]), 0
                else:
                    continue
                ang = _angle(heading, nxt - p)
                if ang < max_turn and (best is None or ang < best[0]):
                    best = (ang, i, new_end)
            if best is None:
                break
            _, cur, end = best
            used.add(cur)
            chosen.append(cur)
    return chosen


def _road_hit(best, roads_m, bound: BaseGeometry, fwd, inv) -> OsmHit | None:
    _, idx, wdt = best
    tags = roads_m[idx][1]
    name = tags.get("name")
    if name:
        group = [i for i, (_, t, _) in enumerate(roads_m) if t.get("name") == name]
    else:
        group = _follow_stroke(idx, roads_m)
    merged_m = shapely.line_merge(unary_union([roads_m[i][0] for i in group]))
    merged = reproject(merged_m, inv).intersection(bound)
    if merged.is_empty:
        return None
    poly_m = reproject(merged, fwd).buffer(wdt / 2, cap_style="round", join_style="round")
    poly = reproject(poly_m, inv).intersection(bound)
    if poly.is_empty:
        return None
    return OsmHit("road", poly, tags, centerline=merged, width_m=wdt,
                  osm_ids=sorted({roads_m[i][2] for i in group}))


def view_box(b: list[float]) -> Polygon:
    return box(*b)
