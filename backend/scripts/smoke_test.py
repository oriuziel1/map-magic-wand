r"""Smoke test: mercator round-trip, mosaic, OSM hit, SAM2 prediction.
Run from backend/:  .venv\Scripts\python.exe -m scripts.smoke_test"""
import asyncio
import time

import numpy as np
from PIL import Image

from app import net  # noqa: F401
from app import osm
from app.sam import ENGINE
from app.tiles import TileStore, lnglat_to_px, px_to_lnglat_arr, window_around
from shapely.geometry import Point

LNG, LAT = 34.7917, 32.0743  # Tel Aviv - Azrieli area


async def main():
    x, y = lnglat_to_px(LNG, LAT, 18)
    back = px_to_lnglat_arr(np.array([[x, y]]), 18)[0]
    assert abs(back[0] - LNG) < 1e-9 and abs(back[1] - LAT) < 1e-9, back
    print("mercator round-trip OK")

    store = TileStore()
    for layer in ("ortho", "street"):
        t = time.time()
        m = await store.mosaic(layer, 18, *window_around(LNG, LAT, 18, 1024))
        Image.fromarray(m.image).save(f"cache/mosaic_{layer}.png")
        print(f"mosaic {layer} {m.image.shape} mean={m.image.mean():.1f} in {time.time()-t:.1f}s")
    await store.close()

    t = time.time()
    bound = Point(LNG, LAT).buffer(0.002)
    areas, roads = await osm.load_features(LNG, LAT, bound.bounds)
    hit = osm.find_hit(LNG, LAT, areas, roads, bound, None)
    print(f"overpass: {len(areas)} areas, {len(roads)} roads in {time.time()-t:.1f}s; hit={hit and (hit.kind, hit.tags)}")

    ENGINE.load()
    print("sam ready:", ENGINE.ready, ENGINE.device, ENGINE.error)
    cx, cy = m.to_local(LNG, LAT)
    for i in range(2):
        t = time.time()
        res = ENGINE.predict(m.key, m.image, [(cx, cy, 1)])
        print(f"predict #{i}: {[(int(mk.sum()), round(s, 3)) for mk, s in res]} in {time.time()-t:.2f}s")


asyncio.run(main())
