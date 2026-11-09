from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cache = Path(cfg.get("cache_dir", "cache"))
    if not cache.is_absolute():
        cache = BASE_DIR / cache
    cfg["cache_dir"] = cache
    return cfg


CONFIG = load_config()
