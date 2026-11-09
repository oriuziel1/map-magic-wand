import type { Feature, MultiPolygon, Polygon } from "geojson";

export type LayerId = "ortho" | "street";
export type Mode = "auto" | "osm" | "ai";
export type Tool = "wand" | "clip" | "edit";

export type WandFeature = Feature<Polygon | MultiPolygon, {
  source: "osm" | "sam2" | "osm+sam2" | "edited";
  area_m2: number;
  kind?: string;
  score?: number;
  osm_tags?: Record<string, string | number>;
  osm_ids?: number[];
}>;

export interface WandRequest {
  lng: number;
  lat: number;
  layer: LayerId;
  zoom: number;
  view_bbox: [number, number, number, number];
  clip: Polygon | null;
  mode: Mode;
  points: [number, number, number][];
  simplify: number;
}

export interface WandResponse {
  feature: WandFeature | null;
  alternatives: WandFeature[];
  selected: number;
  messages: string[];
  debug: { tile_zoom: number; ms: number; device: string };
}

export interface Selection {
  id: number;
  request: WandRequest;
  alternatives: WandFeature[];
  selected: number;
  ms: number;
}

export interface AppConfig {
  center: [number, number];
  zoom: number;
  layers: Record<LayerId, { name: string; max_zoom: number; attribution: string; tiles: string }>;
}

export interface Health {
  sam_ready: boolean;
  sam_loading: boolean;
  sam_error: string | null;
  device: string;
  model: string;
}
