import { useCallback, useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, Map as MLMap, MapMouseEvent } from "maplibre-gl";
import { TerraDraw, TerraDrawPolygonMode, TerraDrawSelectMode } from "terra-draw";
import { TerraDrawMapLibreGLAdapter } from "terra-draw-maplibre-gl-adapter";
import type { FeatureCollection, Polygon, Position } from "geojson";
import { getConfig, getHealth, runWand } from "./api";
import type { AppConfig, Health, LayerId, Mode, Selection, Tool, WandFeature, WandRequest } from "./types";

const SOURCE_COLORS: Record<string, string> = {
  osm: "#3b82f6",
  sam2: "#f59e0b",
  "osm+sam2": "#10b981",
  edited: "#a855f7",
};
const SOURCE_LABEL: Record<string, string> = {
  osm: "OSM vector",
  sam2: "SAM2 (AI)",
  "osm+sam2": "OSM-guided SAM2",
  edited: "Edited",
};
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

interface Toast { id: number; text: string; kind: "info" | "warn" | "error" }

let nextId = 1;

export default function App() {
  const mapDiv = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const drawRef = useRef<TerraDraw | null>(null);
  const suppressClickUntil = useRef(0);

  const [config, setConfig] = useState<AppConfig | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [layer, setLayer] = useState<LayerId>("ortho");
  const [tool, setTool] = useState<Tool>("wand");
  const [mode, setMode] = useState<Mode>("auto");
  const [simplify, setSimplify] = useState(1.5);
  const [clip, setClip] = useState<Polygon | null>(null);
  const [selections, setSelections] = useState<Selection[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [mapReady, setMapReady] = useState(false);

  // refs so map event handlers always see current state
  const st = useRef({ layer, tool, mode, simplify, clip, selections, activeId, busy, editingId });
  st.current = { layer, tool, mode, simplify, clip, selections, activeId, busy, editingId };

  const toast = useCallback((text: string, kind: Toast["kind"] = "info") => {
    const id = nextId++;
    setToasts((t) => [...t, { id, text, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 7000);
  }, []);

  // ------------------------------------------------------------------ boot
  useEffect(() => {
    getConfig().then(setConfig).catch((e) => toast(`Backend unreachable: ${e.message}`, "error"));
    const poll = () => getHealth().then(setHealth).catch(() => setHealth(null));
    poll();
    const h = setInterval(poll, 4000);
    return () => clearInterval(h);
  }, [toast]);

  useEffect(() => {
    if (!config || !mapDiv.current || mapRef.current) return;
    const origin = window.location.origin;
    const map = new maplibregl.Map({
      container: mapDiv.current,
      style: {
        version: 8,
        sources: Object.fromEntries(
          Object.entries(config.layers).map(([id, l]) => [
            id,
            { type: "raster", tiles: [origin + l.tiles], tileSize: 256, maxzoom: l.max_zoom, attribution: l.attribution },
          ]),
        ),
        layers: [
          { id: "ortho", type: "raster", source: "ortho" },
          { id: "street", type: "raster", source: "street", layout: { visibility: "none" } },
        ],
      },
      center: config.center,
      zoom: config.zoom,
      maxZoom: 20,
      boxZoom: false,
      doubleClickZoom: false,
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl({}), "bottom-right");
    mapRef.current = map;
    (window as unknown as { __map: MLMap }).__map = map; // handy for debugging

    map.on("load", () => {
      map.addSource("clip", { type: "geojson", data: EMPTY });
      map.addSource("results", { type: "geojson", data: EMPTY });
      map.addSource("prompts", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "clip-fill", type: "fill", source: "clip", paint: { "fill-color": "#e879f9", "fill-opacity": 0.06 } });
      map.addLayer({
        id: "clip-line", type: "line", source: "clip",
        paint: { "line-color": "#e879f9", "line-width": 2, "line-dasharray": [3, 2] },
      });
      const color: maplibregl.ExpressionSpecification = [
        "match", ["get", "source"],
        ...Object.entries(SOURCE_COLORS).flat(),
        "#ef4444",
      ] as unknown as maplibregl.ExpressionSpecification;
      map.addLayer({
        id: "results-fill", type: "fill", source: "results",
        paint: { "fill-color": color, "fill-opacity": ["case", ["get", "active"], 0.38, 0.2] },
      });
      map.addLayer({
        id: "results-line", type: "line", source: "results",
        paint: { "line-color": color, "line-width": ["case", ["get", "active"], 3, 1.5] },
      });
      map.addLayer({
        id: "prompts", type: "circle", source: "prompts",
        paint: {
          "circle-radius": 5,
          "circle-color": ["case", ["==", ["get", "label"], 1], "#22c55e", "#ef4444"],
          "circle-stroke-color": "#fff", "circle-stroke-width": 1.5,
        },
      });

      const draw = new TerraDraw({
        adapter: new TerraDrawMapLibreGLAdapter({ map, coordinatePrecision: 9 }),
        modes: [
          new TerraDrawPolygonMode({
            styles: { fillColor: "#e879f9", outlineColor: "#e879f9", fillOpacity: 0.1, outlineWidth: 2 },
          }),
          new TerraDrawSelectMode({
            flags: {
              polygon: {
                feature: { draggable: true, coordinates: { midpoints: true, draggable: true, deletable: true } },
              },
            },
            styles: {
              selectedPolygonColor: "#a855f7", selectedPolygonOutlineColor: "#a855f7",
              selectedPolygonFillOpacity: 0.25, selectionPointColor: "#ffffff", midPointColor: "#a855f7",
            },
          }),
        ],
      });
      draw.start();
      draw.on("finish", (id, ctx) => {
        if (ctx.mode === "polygon" && st.current.tool === "clip") {
          const f = draw.getSnapshotFeature(id);
          draw.removeFeatures([id]);
          suppressClickUntil.current = Date.now() + 400; // the closing click must not trigger the wand
          if (f && f.geometry.type === "Polygon") {
            setClip(f.geometry as Polygon);
            toast("Clip area set - wand results are now bounded by it.");
          }
          draw.setMode("static");
          setTool("wand");
        }
      });
      drawRef.current = draw;
      setMapReady(true);
    });

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config]);

  // ------------------------------------------------------------------ wand
  const execute = useCallback(async (req: WandRequest, replaceId: number | null) => {
    setBusy(true);
    try {
      const res = await runWand(req);
      res.messages.forEach((m) => toast(m, "warn"));
      if (!res.feature) {
        if (!res.messages.length) toast("Nothing found here.", "warn");
        return;
      }
      const sel: Selection = {
        id: replaceId ?? nextId++, request: req, alternatives: res.alternatives,
        selected: res.selected, ms: res.debug.ms,
      };
      setSelections((s) => (replaceId ? s.map((x) => (x.id === replaceId ? sel : x)) : [...s, sel]));
      setActiveId(sel.id);
    } catch (e) {
      toast(`Wand failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [toast]);

  const onMapClick = useCallback((e: MapMouseEvent) => {
    const s = st.current;
    const map = mapRef.current;
    if (!map || s.tool !== "wand" || s.busy || Date.now() < suppressClickUntil.current) return;
    const ev = e.originalEvent;
    const active = s.selections.find((x) => x.id === s.activeId);
    if ((ev.shiftKey || ev.altKey) && active) {
      // refine current selection with an extra SAM2 prompt point
      const label = ev.altKey ? 0 : 1;
      const req: WandRequest = {
        ...active.request,
        mode: "ai",
        points: [...active.request.points, [e.lngLat.lng, e.lngLat.lat, label]],
        simplify: s.simplify,
      };
      execute(req, active.id);
      return;
    }
    const b = map.getBounds();
    execute({
      lng: e.lngLat.lng, lat: e.lngLat.lat, layer: s.layer, zoom: map.getZoom(),
      view_bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()],
      clip: s.clip, mode: s.mode, points: [], simplify: s.simplify,
    }, null);
  }, [execute]);

  // keep the click handler fresh (it closes over execute)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.off("click", onMapClick);
    map.on("click", onMapClick);
    return () => { map.off("click", onMapClick); };
  }, [onMapClick, mapReady]);

  // ------------------------------------------------------------------ sync map
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    map.setLayoutProperty("ortho", "visibility", layer === "ortho" ? "visible" : "none");
    map.setLayoutProperty("street", "visibility", layer === "street" ? "visible" : "none");
  }, [layer, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    const src = map?.getSource("results") as GeoJSONSource | undefined;
    if (!src) return;
    src.setData({
      type: "FeatureCollection",
      features: selections
        .filter((s) => s.id !== editingId)
        .map((s) => {
          const f = s.alternatives[s.selected];
          return { ...f, properties: { ...f.properties, active: s.id === activeId, selId: s.id } };
        }),
    });
    const active = selections.find((s) => s.id === activeId);
    const pts = active
      ? [[active.request.lng, active.request.lat, 1], ...active.request.points].filter(() => active.request.points.length > 0)
      : [];
    (map!.getSource("prompts") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: pts.map(([lng, lat, label]) => ({
        type: "Feature", geometry: { type: "Point", coordinates: [lng, lat] }, properties: { label },
      })),
    });
  }, [selections, activeId, editingId, mapReady]);

  useEffect(() => {
    const src = mapRef.current?.getSource("clip") as GeoJSONSource | undefined;
    src?.setData(clip ? { type: "Feature", geometry: clip, properties: {} } : EMPTY);
  }, [clip, mapReady]);

  // ------------------------------------------------------------------ tools
  const finishEdit = useCallback(() => {
    const draw = drawRef.current;
    const id = st.current.editingId;
    if (!draw || id == null) return;
    const f = draw.getSnapshot().find((x) => x.properties?.selId === id);
    draw.clear();
    draw.setMode("static");
    setEditingId(null);
    if (f && f.geometry.type === "Polygon") {
      const geom = f.geometry as Polygon;
      setSelections((ss) => ss.map((s) => {
        if (s.id !== id) return s;
        const edited: WandFeature = {
          type: "Feature", geometry: geom,
          properties: { ...s.alternatives[s.selected].properties, source: "edited", area_m2: Math.round(polygonArea(geom.coordinates)) },
        };
        return { ...s, alternatives: [...s.alternatives, edited], selected: s.alternatives.length };
      }));
    }
  }, []);

  const changeTool = useCallback((t: Tool) => {
    const draw = drawRef.current;
    if (!draw) return;
    if (st.current.editingId != null) finishEdit();
    if (t === "clip") {
      draw.setMode("polygon");
    } else if (t === "edit") {
      const s = st.current.selections.find((x) => x.id === st.current.activeId);
      if (!s) { toast("Select a result first.", "warn"); return; }
      const f = s.alternatives[s.selected];
      let poly: Position[][];
      if (f.geometry.type === "Polygon") poly = f.geometry.coordinates;
      else {
        // edit the largest part of a multipolygon
        poly = f.geometry.coordinates.reduce((a, b) => (polygonArea(b) > polygonArea(a) ? b : a));
        toast("Multi-part result - editing its largest part.", "warn");
      }
      const ring = poly[0].map(([x, y]) => [round(x), round(y)]);
      const res = draw.addFeatures([{
        type: "Feature", geometry: { type: "Polygon", coordinates: [ring] },
        properties: { mode: "polygon", selId: s.id },
      }]);
      if (!res[0]?.valid) { toast(`Cannot edit this shape (${res[0]?.reason ?? "invalid"}).`, "error"); return; }
      draw.setMode("select");
      draw.selectFeature(res[0].id!);
      setEditingId(s.id);
    } else {
      draw.setMode("static");
    }
    setTool(t);
  }, [finishEdit, toast]);

  const cycle = useCallback((id: number, dir: number) => {
    setSelections((ss) => ss.map((s) => s.id === id
      ? { ...s, selected: (s.selected + dir + s.alternatives.length) % s.alternatives.length }
      : s));
  }, []);

  const remove = useCallback((id: number) => {
    setSelections((ss) => ss.filter((s) => s.id !== id));
    setActiveId((a) => (a === id ? null : a));
  }, []);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      const s = st.current;
      if (e.key === "Tab" && s.activeId != null) { e.preventDefault(); cycle(s.activeId, e.shiftKey ? -1 : 1); }
      else if ((e.key === "Delete" || e.key === "Backspace") && s.activeId != null && s.editingId == null) remove(s.activeId);
      else if (e.key === "z" && (e.ctrlKey || e.metaKey) && s.selections.length && s.editingId == null)
        remove(s.selections[s.selections.length - 1].id);
      else if (e.key === "Escape" || (e.key === "Enter" && s.editingId != null)) changeTool("wand");
      else if (e.key === "w") changeTool("wand");
      else if (e.key === "c") changeTool("clip");
      else if (e.key === "e") changeTool("edit");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cycle, remove, changeTool]);

  const exportGeoJSON = () => {
    const fc: FeatureCollection = {
      type: "FeatureCollection",
      features: selections.map((s) => s.alternatives[s.selected]),
    };
    const blob = new Blob([JSON.stringify(fc, null, 2)], { type: "application/geo+json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "magic-wand.geojson";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // ------------------------------------------------------------------ render
  const healthText = !health ? "backend offline"
    : health.sam_ready ? `SAM2 ready · ${health.device}`
    : health.sam_error ? "SAM2 failed (OSM only)" : "SAM2 loading…";
  const healthClass = !health ? "bad" : health.sam_ready ? "ok" : health.sam_error ? "bad" : "wait";

  return (
    <div className="app">
      <aside className="panel">
        <header>
          <h1>✨ Map Magic Wand</h1>
          <span className={`badge ${healthClass}`} title={health?.sam_error ?? health?.model ?? ""}>{healthText}</span>
        </header>

        <section>
          <label className="lbl">Base map</label>
          <div className="seg">
            {(["ortho", "street"] as LayerId[]).map((l) => (
              <button key={l} className={layer === l ? "on" : ""} onClick={() => setLayer(l)}>
                {config?.layers[l]?.name ?? l}
              </button>
            ))}
          </div>
        </section>

        <section>
          <label className="lbl">Tool</label>
          <div className="seg">
            <button className={tool === "wand" ? "on" : ""} onClick={() => changeTool("wand")} title="W">✨ Wand</button>
            <button className={tool === "clip" ? "on" : ""} onClick={() => changeTool("clip")} title="C">⬠ Clip area</button>
            <button className={tool === "edit" ? "on" : ""} onClick={() => changeTool("edit")} title="E">✏️ Edit</button>
          </div>
          {tool === "clip" && <p className="hint">Click to add vertices, click the first point to close.</p>}
          {tool === "edit" && <p className="hint">Drag vertices / midpoints. Press Enter or pick another tool to finish.</p>}
          {clip && (
            <div className="clipinfo">
              <span>Clip area active ({Math.round(polygonArea(clip.coordinates)).toLocaleString()} m²)</span>
              <button className="link" onClick={() => setClip(null)}>clear</button>
            </div>
          )}
        </section>

        <section>
          <label className="lbl">Strategy</label>
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="auto">Auto - OSM vectors first, SAM2 fallback</option>
            <option value="osm">OSM vectors only</option>
            <option value="ai">SAM2 (AI) only</option>
          </select>
          <label className="lbl">Simplify: {simplify.toFixed(1)} px</label>
          <input type="range" min={0} max={6} step={0.5} value={simplify} onChange={(e) => setSimplify(+e.target.value)} />
        </section>

        <section className="list">
          <label className="lbl">Selections ({selections.length})</label>
          {selections.length === 0 && <p className="hint">Click anything on the map with the wand.</p>}
          {selections.map((s) => {
            const f = s.alternatives[s.selected];
            const p = f.properties;
            return (
              <div key={s.id} className={`item ${s.id === activeId ? "active" : ""}`} onClick={() => setActiveId(s.id)}>
                <span className="dot" style={{ background: SOURCE_COLORS[p.source] }} />
                <div className="meta">
                  <div className="title">{describe(f)}</div>
                  <div className="sub">
                    {SOURCE_LABEL[p.source]} · {p.area_m2.toLocaleString()} m²
                    {p.score != null && ` · score ${p.score}`} · {s.ms} ms
                  </div>
                </div>
                {s.alternatives.length > 1 && (
                  <div className="alts" onClick={(e) => e.stopPropagation()}>
                    <button onClick={() => cycle(s.id, -1)}>‹</button>
                    <span>{s.selected + 1}/{s.alternatives.length}</span>
                    <button onClick={() => cycle(s.id, 1)}>›</button>
                  </div>
                )}
                <button className="x" onClick={(e) => { e.stopPropagation(); remove(s.id); }}>×</button>
              </div>
            );
          })}
        </section>

        <section className="row">
          <button disabled={!selections.length} onClick={exportGeoJSON}>⬇ Export GeoJSON</button>
          <button disabled={!selections.length} onClick={() => { setSelections([]); setActiveId(null); }}>Clear all</button>
        </section>

        <footer>
          <b>Click</b> new selection · <b>Shift+click</b> add area · <b>Alt+click</b> remove area ·{" "}
          <b>Tab</b> grow/shrink (cycle alternatives) · <b>Del</b> delete · <b>Ctrl+Z</b> undo
          <div className="legend">
            {Object.entries(SOURCE_LABEL).map(([k, v]) => (
              <span key={k}><i style={{ background: SOURCE_COLORS[k] }} />{v}</span>
            ))}
          </div>
        </footer>
      </aside>

      <main className={`map tool-${tool}`}>
        <div ref={mapDiv} className="mapdiv" />
        {busy && <div className="spinner"><div /> Outlining…</div>}
        <div className="toasts">
          {toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>)}
        </div>
      </main>
    </div>
  );
}

// ------------------------------------------------------------------ helpers
function round(v: number) {
  return Math.round(v * 1e7) / 1e7;
}

function describe(f: WandFeature): string {
  const t = f.properties.osm_tags ?? {};
  if (t.name) return String(t.name);
  if (t.building) return "Building";
  if (t.highway) return `Road (${t.highway})`;
  for (const k of ["landuse", "natural", "leisure", "water", "amenity", "aeroway"]) if (t[k]) return `${k}: ${t[k]}`;
  return f.properties.kind === "road" ? "Road" : "AI segment";
}

/** Geodesic polygon area in m² (rings: outer first, then holes). */
function polygonArea(rings: Position[][]): number {
  const R = 6378137;
  const ringArea = (c: Position[]) => {
    let a = 0;
    for (let i = 0; i < c.length - 1; i++) {
      const [x1, y1] = c[i], [x2, y2] = c[i + 1];
      a += ((x2 - x1) * Math.PI / 180) * (2 + Math.sin(y1 * Math.PI / 180) + Math.sin(y2 * Math.PI / 180));
    }
    return Math.abs(a * R * R / 2);
  };
  return rings.reduce((sum, r, i) => sum + (i === 0 ? ringArea(r) : -ringArea(r)), 0);
}
