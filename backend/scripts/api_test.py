r"""Hit /api/wand for a few targets. Run from backend/ with the server up:
.venv\Scripts\python.exe -m scripts.api_test"""
import json
import httpx
import numpy as np
from shapely.geometry import box, shape

from app.tiles import px_to_lnglat_arr, window_around

LNG, LAT = 34.7917, 32.0743
x0, y0, _, _ = window_around(LNG, LAT, 18, 1024)


def ll(px, py):
    return px_to_lnglat_arr(np.array([[x0 + px, y0 + py]]), 18)[0].tolist()


view = [*ll(0, 1024), *ll(1024, 0)]  # w,s,e,n
clip_box = box(*ll(500, 380), *ll(720, 120))  # a box across the diagonal road

cases = {
    "building auto (ortho)": dict(pt=ll(310, 345), layer="ortho", mode="auto"),
    "building ai (ortho)": dict(pt=ll(310, 345), layer="ortho", mode="ai"),
    "road auto+clip (ortho)": dict(pt=ll(600, 250), layer="ortho", mode="auto", clip=clip_box),
    "road auto+clip (street)": dict(pt=ll(600, 250), layer="street", mode="auto", clip=clip_box),
    "road auto no clip (ortho)": dict(pt=ll(300, 760), layer="ortho", mode="auto"),
    "road area (ortho)": dict(pt=ll(450, 420), layer="ortho", mode="auto"),
    "parking lot ai (ortho)": dict(pt=ll(880, 40), layer="ortho", mode="ai"),
}
out = {}
for name, c in cases.items():
    body = {"lng": c["pt"][0], "lat": c["pt"][1], "layer": c["layer"], "mode": c["mode"], "zoom": 17,
            "view_bbox": view}
    if "clip" in c:
        body["clip"] = c["clip"].__geo_interface__
    r = httpx.post("http://localhost:8000/api/wand", json=body, timeout=120).json()
    f = r["feature"]
    info = None
    if f:
        g = shape(f["geometry"])
        p = f["properties"]
        info = dict(source=p["source"], type=g.geom_type, area=p["area_m2"], verts=len(g.exterior.coords) if g.geom_type == "Polygon" else "multi",
                    tags=p.get("osm_tags"), within_clip=(g.within(c["clip"].buffer(1e-7)) if "clip" in c else None))
        out[name] = f
    print(f"{name}: {info} alts={len(r['alternatives'])} msgs={r['messages']} {r['debug']}")
json.dump({"type": "FeatureCollection", "features": list(out.values())}, open("cache/api_test.geojson", "w"))

# ---- render results over the ortho mosaic for eyeballing
from PIL import Image, ImageDraw  # noqa: E402

img = Image.open("cache/mosaic_ortho.png").convert("RGB")
d = ImageDraw.Draw(img, "RGBA")
colors = {"osm": (59, 130, 246), "sam2": (245, 158, 11), "osm+sam2": (16, 185, 129)}
from app.tiles import lnglat_to_px_arr  # noqa: E402


def px(coords):
    a = lnglat_to_px_arr(np.array(coords), 18) - np.array([x0, y0])
    return [tuple(p) for p in a]


d.line(px(list(clip_box.exterior.coords)), fill=(255, 0, 255, 255), width=2)
for f in out.values():
    g = shape(f["geometry"])
    col = colors[f["properties"]["source"]]
    for poly in getattr(g, "geoms", [g]):
        d.polygon(px(list(poly.exterior.coords)), fill=col + (90,), outline=col + (255,), width=3)
img.save("cache/api_test.png")
print("rendered cache/api_test.png")
