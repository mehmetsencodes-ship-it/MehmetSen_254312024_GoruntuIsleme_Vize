"""Veri seti okuma yardımcıları.

TIFF dosyaları LZW ile sıkıştırılmıştır. Ek paket (imagecodecs) gerektirmemek ve
Windows'ta Türkçe karakter içeren yollarda sorun yaşamamak için dosyalar
np.fromfile + cv2.imdecode ile okunur.
"""
from pathlib import Path
import zipfile

import cv2
import numpy as np

DATASET_NAME = "Fluo-N2DH-GOWT1"
DATASET_URL = "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip"


def ensure_dataset(data_dir, zip_path=None):
    """Veri seti klasörünü döndürür; yoksa ve zip verilmişse zip'i açar."""
    data_dir = Path(data_dir)
    root = data_dir / DATASET_NAME
    if root.exists():
        return root
    if zip_path is not None and Path(zip_path).exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        return root
    raise FileNotFoundError(
        f"{root} bulunamadı. Veri setini {DATASET_URL} adresinden indirip "
        f"'{data_dir}' klasörüne açın veya zip yolunu belirtin."
    )


def read_tif(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Görüntü okunamadı: {path}")
    return img


def image_path(root, sequence, frame):
    return Path(root) / sequence / f"t{frame:03d}.tif"


def tra_path(root, sequence, frame):
    return Path(root) / f"{sequence}_GT" / "TRA" / f"man_track{frame:03d}.tif"


def list_frames(root, sequence):
    return sorted(int(p.stem[1:]) for p in (Path(root) / sequence).glob("t*.tif"))


def load_image(root, sequence, frame):
    return read_tif(image_path(root, sequence, frame))


def reference_count(root, sequence, frame):
    """TRA etiket görüntüsündeki 0 dışındaki farklı hücre kimliklerinin sayısı."""
    path = tra_path(root, sequence, frame)
    if not path.exists():
        return None
    labels = read_tif(path)
    ids = np.unique(labels)
    return int(np.count_nonzero(ids))
