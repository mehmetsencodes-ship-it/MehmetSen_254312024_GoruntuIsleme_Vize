"""Seed tabanlı, deterministik geliştirme / doğrulama / test kare seçimi.

Kurallar (ödev Bölüm 3):
  * Sequence 01'den 5 doğrulama karesi, ardından aynı sekanstan 10 geliştirme karesi.
  * Geliştirme ve doğrulama kareleri arasında en az GAP karelik boşluk:
    |f_dev - f_val| > GAP  (yani iki kare arasında en az GAP kare bulunur).
  * Sequence 02'den 3 başlangıç noktası, her birinden ardışık 5 kare.
  * Test blokları arasında en az GAP karelik boşluk:
    start_{i+1} - (start_i + BLOCK_LEN - 1) > GAP.
  * Koşullar sağlanamazsa aynı RNG akışından (aynı seed) yeniden çekim yapılır;
    deneme sayısı kaydedilir. Bu prosedür tamamen deterministiktir.
"""
import numpy as np
import pandas as pd

N_VAL, N_DEV = 5, 10
N_BLOCKS, BLOCK_LEN = 3, 5
GAP = 5
MAX_ATTEMPTS = 10_000


def _select_dev_val(frames, rng):
    frames = np.asarray(frames)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        val = np.sort(rng.choice(frames, N_VAL, replace=False))
        candidates = np.array([f for f in frames if np.all(np.abs(f - val) > GAP)])
        if len(candidates) >= N_DEV:
            dev = np.sort(rng.choice(candidates, N_DEV, replace=False))
            return dev.tolist(), val.tolist(), attempt
    raise RuntimeError("Geliştirme/doğrulama seçimi koşulları sağlanamadı.")


def _select_test_blocks(frames, rng):
    frames = np.asarray(frames)
    possible_starts = frames[: len(frames) - BLOCK_LEN + 1]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        starts = np.sort(rng.choice(possible_starts, N_BLOCKS, replace=False))
        if np.all(np.diff(starts) - (BLOCK_LEN - 1) > GAP):
            return [list(range(s, s + BLOCK_LEN)) for s in starts], attempt
    raise RuntimeError("Test bloğu seçimi koşulları sağlanamadı.")


def select_frames(frames_01, frames_02, seed):
    """Seçimi yapar ve selected_frames.csv biçiminde bir DataFrame döndürür."""
    rng = np.random.default_rng(seed)
    dev, val, att_dv = _select_dev_val(frames_01, rng)
    blocks, att_t = _select_test_blocks(frames_02, rng)

    rows = []
    for f in dev:
        rows.append(("development", "01", f, None, att_dv))
    for f in val:
        rows.append(("validation", "01", f, None, att_dv))
    for b, block in enumerate(blocks, start=1):
        for f in block:
            rows.append(("test", "02", f, b, att_t))
    df = pd.DataFrame(rows, columns=["group", "sequence", "frame", "test_block", "selection_attempt"])
    df["file_name"] = df.apply(lambda r: f"{r.sequence}/t{r.frame:03d}.tif", axis=1)
    df["seed"] = seed
    df["test_block"] = df["test_block"].astype("Int64")
    return df[["seed", "group", "sequence", "frame", "file_name", "test_block", "selection_attempt"]]


def check_selection(df):
    """Seçim kurallarını doğrular; ihlal varsa AssertionError fırlatır."""
    dev = df.query("group == 'development'").frame.to_numpy()
    val = df.query("group == 'validation'").frame.to_numpy()
    assert len(dev) == N_DEV and len(val) == N_VAL
    assert not set(dev) & set(val)
    assert np.all(np.abs(dev[:, None] - val[None, :]) > GAP)
    test = df.query("group == 'test'")
    assert len(test) == N_BLOCKS * BLOCK_LEN
    starts = sorted(test.groupby("test_block").frame.min())
    for b, g in test.groupby("test_block"):
        fr = np.sort(g.frame.to_numpy())
        assert np.all(np.diff(fr) == 1) and len(fr) == BLOCK_LEN
    assert np.all(np.diff(starts) - (BLOCK_LEN - 1) > GAP)
    return True
