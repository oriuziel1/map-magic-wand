# Map Magic Wand: POC

Click something on a map (orthophoto or street map) and get a polygon around it. For long things like roads, draw a **clip area** first and the result stays inside it.

## How it works

```
click ─► 1. OSM vectors      is there a mapped building / road / small area under the click?
         │   ├─ building / area ............ return the real OSM footprint            (blue)
         │   └─ road ......................... follow the road "stroke" through junctions,
         │                                      buffer by its width, clip to the area / view
         │        └─ on orthophoto ........... use the centerline as SAM2 prompt points so the
         │                                      outline follows the real pavement   (green)
         └─ 2. nothing mapped ─► SAM2 on the imagery under the click, mask → polygon (orange)
```

* **SAM2.1** (`facebook/sam2.1-hiera-base-plus`, via HF `transformers`) runs locally on the GPU. Image embeddings are cached per tile window, so the first click in an area takes about 0.5–1 s and later ones take about 20–50 ms.
* **OSM data** comes from OpenMapTiles-schema **vector tiles** (OpenFreeMap by default), which are much faster and more reliable than the public Overpass API. Overpass is still supported (`osm.provider: overpass`) and gives richer tags and road names.
* Large OSM areas (more than 20,000 m², such as neighbourhoods or landuse) are skipped in *Auto* mode, so the wand outlines what is really under the cursor. Use *OSM only* mode to get them.
* SAM2 returns 3 masks (sub-part, part, whole). Press **Tab** to cycle through them ("grow/shrink").

## Run

Prerequisites: Python 3.12+ (tested on 3.14), Node 20+, NVIDIA GPU recommended (CPU works but is slow).

```bash
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api to :8000)
```

The SAM2 weights (about 320 MB) download from Hugging Face on first start. The UI badge turns green once the model is loaded. Behind a TLS-inspecting proxy, Python uses the Windows cert store (`truststore`). For `uv`, set `UV_SYSTEM_CERTS=1`.

## Use

| Action | |
|---|---|
| **Click** | new selection |
| **Shift+click / Alt+click** | add / remove a region from the current selection (extra SAM2 prompt points) |
| **Tab / Shift+Tab** | cycle alternatives (SAM2 masks, OSM vs OSM-guided SAM2, edited) |
| **C** / Clip area | draw a bounding polygon; later results are clipped to it |
| **E** / Edit | drag vertices / midpoints of the active result; Enter to finish |
| **Del**, **Ctrl+Z** | delete active / last selection |
| Export GeoJSON | download all selections with their properties (source, area, OSM tags, score) |

The strategy dropdown can force *OSM only* or *SAM2 only*.

## Going offline

Everything external is configured in [backend/config.yaml](backend/config.yaml):

| What | Online default | Offline replacement |
|---|---|---|
| `layers.*.url` | Esri World Imagery / OSM tiles | your orthophoto as XYZ (e.g. GeoTIFF → MapProxy / TiTiler / gdal2tiles) |
| `osm.vector_tiles.url` | OpenFreeMap TileJSON | a region `.mbtiles` (OpenMapTiles schema) served by tileserver-gl or martin |
| `sam.model` | HF model id | a local folder with the downloaded model |

Tiles, vector tiles and Overpass responses are also cached on disk in `backend/cache/`, so an area you have already looked at works without network.

## Layout

```
backend/app/
  main.py      FastAPI: /api/config, /api/health, /api/tiles/{layer}/{z}/{x}/{y} (caching proxy), POST /api/wand
  wand.py      strategy: OSM first → SAM2 fallback / OSM-guided SAM2 for roads
  osm.py       vector-tile + Overpass providers, hit-testing, road stroke following, buffering
  sam.py       SAM2 wrapper with embedding cache
  tiles.py     web-mercator math, XYZ fetch + disk cache, image mosaics
  geometry.py  mask → polygon (holes, gap closing, simplify), clipping, areas
backend/scripts/smoke_test.py  model/tiles/OSM sanity check
backend/scripts/api_test.py    end-to-end /api/wand cases + rendered overlay (cache/api_test.png)
frontend/src/App.tsx           React + MapLibre GL + terra-draw UI
```

## Known limitations / next steps

* Imagery and OSM can be misaligned by a few metres (and tall buildings lean in orthophotos). SAM2 mode follows the pixels; OSM mode follows the map.
* A road selection without a clip area is bounded by the current view. Very long roads at low zoom lose detail because the mosaic is capped at 2048 px. Tiled SAM2 runs along the centerline would fix that.
* Vector tiles carry no road names, so roads are grouped by "straightest continuation of the same class". Overpass mode groups by name.
* Possible improvements: SAM2 box prompts from a drag, automatic negative points on neighbouring buildings, a class-aware post-processor (squaring building corners), and a fine-tuned remote-sensing SAM (e.g. SAMGeo / RSPrompter weights).
