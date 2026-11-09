import type { AppConfig, Health, WandRequest, WandResponse } from "./types";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const getConfig = () => fetch("/api/config").then((r) => json<AppConfig>(r));
export const getHealth = () => fetch("/api/health").then((r) => json<Health>(r));
export const runWand = (req: WandRequest) =>
  fetch("/api/wand", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => json<WandResponse>(r));
