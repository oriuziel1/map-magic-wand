"""SAM2 wrapper (HuggingFace transformers) with an image-embedding cache."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

import numpy as np
from PIL import Image

from .config import CONFIG

log = logging.getLogger("sam")


class SamEngine:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.error: str | None = None
        self.loading = False
        self._lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()  # mosaic key -> image embeddings

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        self.loading = True
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor

            cfg = CONFIG["sam"]
            dev = cfg.get("device", "auto")
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
            t = time.time()
            self.processor = Sam2Processor.from_pretrained(cfg["model"])
            self.model = Sam2Model.from_pretrained(cfg["model"]).to(self.device).eval()
            log.info("SAM2 %s loaded on %s in %.1fs", cfg["model"], self.device, time.time() - t)
        except Exception as ex:  # keep the server alive; OSM path still works
            self.error = f"{ex.__class__.__name__}: {ex}"
            log.exception("SAM2 failed to load")
        finally:
            self.loading = False

    def _embeddings(self, key, pil: Image.Image):
        import torch

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self.model.get_image_embeddings(inputs["pixel_values"])
        self._cache[key] = emb
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return emb

    def predict(self, key, image: np.ndarray, points: list[tuple[float, float, int]],
                box: tuple[float, float, float, float] | None = None,
                multimask: bool = True) -> list[tuple[np.ndarray, float]]:
        """points: (x, y, label) in image pixels, label 1=positive 0=negative.
        Returns [(mask HxW bool, score)] sorted best-first."""
        import torch

        if not self.ready:
            raise RuntimeError(self.error or "SAM2 model is still loading")
        with self._lock:
            pil = Image.fromarray(image)
            emb = self._embeddings(key, pil)
            kw = {}
            if points:
                kw["input_points"] = [[[[float(x), float(y)] for x, y, _ in points]]]
                kw["input_labels"] = [[[int(l) for _, _, l in points]]]
            if box is not None:
                kw["input_boxes"] = [[[float(v) for v in box]]]
            inputs = self.processor(images=pil, return_tensors="pt", **kw).to(self.device)
            inputs.pop("pixel_values", None)
            with torch.inference_mode():
                out = self.model(**inputs, image_embeddings=emb, multimask_output=multimask)
            masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
            masks = masks[0].numpy().astype(bool)  # (n_masks, H, W) for object 0
            scores = out.iou_scores[0, 0].float().cpu().numpy().tolist()
        res = list(zip(masks, scores))
        res.sort(key=lambda t: -t[1])
        return res


ENGINE = SamEngine()
