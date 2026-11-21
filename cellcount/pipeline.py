"""JSON ile tanımlanabilen görüntü işleme pipeline'ı.

Örnek konfigürasyon:
{
  "name": "Pipeline B",
  "intensity": {"method": "stretch", "p_low": 1, "p_high": 99.5},
  "filter": [{"method": "gaussian", "ksize": 5}],
  "segmentation": {"method": "otsu"},
  "morphology": [{"op": "open", "ksize": 5}, {"op": "fill_holes"}],
  "separation": {"method": "watershed_peaks", "min_distance": 15},
  "min_area": 150
}
"""
import copy
import json
import time

import numpy as np

from . import ops

DEFAULT = {
    "name": "unnamed",
    "intensity": {"method": "none"},
    "filter": [],
    "segmentation": {"method": "otsu"},
    "morphology": [],
    "separation": {"method": "none"},
    "min_area": 0,
}

BASELINE_NAME = "Baseline (Otsu + CCL)"


def make_config(**kw):
    cfg = copy.deepcopy(DEFAULT)
    cfg.update(copy.deepcopy(kw))
    return cfg


def baseline_config(min_area):
    """Ödev Bölüm 5: gri görüntü -> Otsu -> bağlı bileşen -> sayım (+ sabit min alan)."""
    return make_config(name=BASELINE_NAME, segmentation={"method": "otsu"}, min_area=min_area)


def run_pipeline(img, cfg, reference=None, return_stages=False):
    """Görüntüyü işler; (sayım, etiketler, aşamalar) döndürür."""
    stages = {}
    x = ops.to_uint8(img)
    stages["original"] = x
    x = ops.apply_intensity(x, cfg.get("intensity", {"method": "none"}), reference)
    stages["intensity"] = x
    for f in cfg.get("filter", []):
        x = ops.apply_filter(x, f)
    stages["filtered"] = x
    mask = ops.segment(x, cfg["segmentation"])
    stages["mask"] = mask
    mask = ops.apply_morphology(mask, cfg.get("morphology", []))
    stages["morph"] = mask
    labels, dist = ops.separate(mask, cfg.get("separation", {"method": "none"}))
    if dist is not None:
        stages["distance"] = dist
    labels, count = ops.filter_by_area(labels, cfg.get("min_area", 0), cfg.get("max_area"))
    stages["labels"] = labels
    return (count, labels, stages) if return_stages else (count, labels, None)


def evaluate_config(cfg, images, references=None, reference_img=None):
    """Bir konfigürasyonu görüntü sözlüğü {frame: img} üzerinde çalıştırır.

    references verilirse hata metrikleri hesaplanır; test tahmininde verilmez.
    """
    rows = []
    for frame, img in images.items():
        t0 = time.perf_counter()
        count, _, _ = run_pipeline(img, cfg, reference_img)
        dt = time.perf_counter() - t0
        row = {"frame": frame, "pred": count, "time_s": dt}
        if references is not None:
            row["ref"] = references[frame]
            row["error"] = count - references[frame]
        rows.append(row)
    return rows


def describe(cfg):
    """Deney tablosu için okunabilir sütunlar üretir."""
    def fmt(d, skip=("method", "op")):
        extra = ",".join(f"{k}={v}" for k, v in d.items() if k not in skip)
        return extra

    it = cfg["intensity"]
    intensity = it["method"] + (f"({fmt(it)})" if fmt(it) else "")
    filt = " -> ".join(f["method"] + (f"({fmt(f)})" if fmt(f) else "") for f in cfg["filter"]) or "none"
    sg = cfg["segmentation"]
    seg = sg["method"] + (f"({fmt(sg)})" if fmt(sg) else "")
    morph = " -> ".join(m["op"] + (f"({m['ksize']})" if "ksize" in m else "") for m in cfg["morphology"]) or "none"
    sp = cfg["separation"]
    sep = sp["method"] + (f"({fmt(sp)})" if fmt(sp) else "")
    return {"intensity": intensity, "filter": filt, "segmentation": seg, "morphology": morph,
            "separation": sep, "min_area": cfg["min_area"]}


def save_config(cfg, path, extra=None):
    data = {"pipeline": cfg}
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))
