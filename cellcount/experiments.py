"""Geliştirme kümesinde kontrollü (tek faktör değiştirilen) deneyler.

Protokol: Aşamalı (greedy / coordinate-descent) arama.
  S0  Baseline min-alan eşiği   (Otsu + CCL; yalnız min_area değişir)
  S1  Yoğunluk dönüşümü         (diğer her şey referans konfigürasyonda sabit)
  S2  Uzamsal filtre            (S1'in en iyisi sabit)
  S3  Segmentasyon              (S1-S2'nin en iyileri sabit)
  S4  Morfoloji / işlem sırası  (S1-S3 sabit)
  S5  Temas eden hücre ayrıştırma (distance transform + watershed)
  S6  Minimum nesne alanı

Her aşamada bir önceki aşamada seçilen konfigürasyondan yalnızca ilgili
bileşen değiştirilir. Aşama kazananı geliştirme MAE'sine göre, eşitlikte
RMSE ve ardından ortalama süre ile belirlenir. Test verisi hiçbir aşamada
kullanılmaz.
"""
import copy

import pandas as pd

from .metrics import count_metrics
from .pipeline import baseline_config, describe, evaluate_config, make_config

REFERENCE = make_config(
    name="reference",
    intensity={"method": "none"},
    filter=[{"method": "gaussian", "ksize": 5}],
    segmentation={"method": "otsu"},
    morphology=[{"op": "open", "ksize": 5}, {"op": "fill_holes"}],
    separation={"method": "none"},
    min_area=400,
)

BASELINE_MIN_AREAS = [0, 25, 50, 100, 200, 400]

INTENSITY_OPTIONS = [
    {"method": "none"},
    {"method": "stretch", "p_low": 1.0, "p_high": 99.5},
    {"method": "gamma", "gamma": 0.5, "p_high": 99.5},
    {"method": "gamma", "gamma": 0.3, "p_high": 99.5},
    {"method": "log", "p_high": 99.5},
    {"method": "hist_eq"},
    {"method": "clahe", "clip_limit": 2.0, "tile": 8},
    {"method": "clahe", "clip_limit": 8.0, "tile": 8},
    {"method": "hist_match"},
]

FILTER_OPTIONS = [
    [],
    [{"method": "gaussian", "ksize": 3}],
    [{"method": "gaussian", "ksize": 5}],
    [{"method": "gaussian", "ksize": 9}],
    [{"method": "gaussian", "ksize": 15}],
    [{"method": "median", "ksize": 5}],
    [{"method": "mean", "ksize": 5}],
    [{"method": "sharpen", "kernel": "laplacian4"}],
    [{"method": "gaussian", "ksize": 5}, {"method": "sharpen", "kernel": "laplacian4"}],
    [{"method": "unsharp", "ksize": 9, "amount": 1.0}],
]

SEGMENTATION_OPTIONS = (
    [{"method": "fixed", "threshold": t} for t in (40, 80, 120, 160)]
    + [
        {"method": "otsu"},
        {"method": "triangle"},
        {"method": "adaptive", "block_size": 151, "C": -5, "floor": 30},
        {"method": "adaptive", "block_size": 301, "C": -5, "floor": 30},
        {"method": "kmeans", "k": 3, "n_background": 1},
    ]
)

_FILL = {"op": "fill_holes"}
MORPHOLOGY_OPTIONS = [
    [],
    [_FILL],
    [{"op": "erode", "ksize": 5}, _FILL],
    [{"op": "dilate", "ksize": 5}, _FILL],
    [{"op": "open", "ksize": 3}, _FILL],
    [{"op": "open", "ksize": 5}, _FILL],
    [{"op": "open", "ksize": 9}, _FILL],
    [{"op": "open", "ksize": 15}, _FILL],
    [{"op": "close", "ksize": 9}, _FILL],
    [{"op": "open", "ksize": 9}, {"op": "close", "ksize": 9}, _FILL],
    [{"op": "close", "ksize": 9}, {"op": "open", "ksize": 9}, _FILL],
    [_FILL, {"op": "open", "ksize": 9}],
]

SEPARATION_OPTIONS = [
    {"method": "none"},
    {"method": "watershed_peaks", "min_distance": 15, "min_peak": 3},
    {"method": "watershed_peaks", "min_distance": 25, "min_peak": 3},
    {"method": "watershed_peaks", "min_distance": 35, "min_peak": 3},
    {"method": "watershed_hmax", "h": 2.0},
    {"method": "watershed_hmax", "h": 4.0},
    {"method": "watershed_hmax", "h": 4.0, "dist_blur": 9},
]

MIN_AREA_OPTIONS = [100, 200, 400, 600, 800]

STAGES = [
    ("S1-intensity", "intensity", INTENSITY_OPTIONS),
    ("S2-filter", "filter", FILTER_OPTIONS),
    ("S3-segmentation", "segmentation", SEGMENTATION_OPTIONS),
    ("S4-morphology", "morphology", MORPHOLOGY_OPTIONS),
    ("S5-separation", "separation", SEPARATION_OPTIONS),
    ("S6-min_area", "min_area", MIN_AREA_OPTIONS),
]


def evaluate(cfg, images, refs, reference_img=None):
    rows = evaluate_config(cfg, images, refs, reference_img)
    m = count_metrics([r["pred"] for r in rows], [r["ref"] for r in rows], [r["time_s"] for r in rows])
    m["preds"] = " ".join(str(r["pred"]) for r in rows)
    return m


def _row(exp_id, stage, changed, cfg, m):
    return {"exp_id": exp_id, "stage": stage, "changed": changed, **describe(cfg),
            "dev_MAE": m["MAE"], "dev_RMSE": m["RMSE"], "dev_Bias": m["Bias"],
            "dev_ExactAcc": m["ExactAcc"], "mean_time_ms": 1000 * m["mean_time_s"],
            "dev_preds": m["preds"], "config": cfg}


def _best(rows):
    return min(rows, key=lambda r: (round(r["dev_MAE"], 6), round(r["dev_RMSE"], 6), r["mean_time_ms"]))


def run_baseline_stage(images, refs):
    rows = []
    for i, a in enumerate(BASELINE_MIN_AREAS, start=1):
        cfg = baseline_config(a)
        rows.append(_row(f"B{i:02d}", "S0-baseline", f"min_area={a}", cfg, evaluate(cfg, images, refs)))
    best = _best(rows)
    for r in rows:
        r["selected"] = r is best
    return rows, best["config"]


def run_greedy_search(images, refs, reference_img, start=REFERENCE, log=print):
    rows, current, n = [], copy.deepcopy(start), 0
    for stage, key, options in STAGES:
        stage_rows = []
        for opt in options:
            n += 1
            cfg = copy.deepcopy(current)
            cfg[key] = copy.deepcopy(opt)
            cfg["name"] = f"E{n:02d}"
            r = _row(f"E{n:02d}", stage, key, cfg, evaluate(cfg, images, refs, reference_img))
            stage_rows.append(r)
            if log:
                log(f"{r['exp_id']} {stage:16s} MAE={r['dev_MAE']:.2f} RMSE={r['dev_RMSE']:.2f} "
                    f"Bias={r['dev_Bias']:+.2f} t={r['mean_time_ms']:.0f}ms  {describe(cfg)[key] if key != 'min_area' else opt}")
        best = _best(stage_rows)
        for r in stage_rows:
            r["selected"] = r is best
        current = copy.deepcopy(best["config"])
        rows += stage_rows
    return rows, current


def best_in_stage(rows, stage, predicate=lambda r: True):
    return _best([r for r in rows if r["stage"] == stage and predicate(r)])


def to_dataframe(rows):
    df = pd.DataFrame(rows).drop(columns=["config"])
    return df
