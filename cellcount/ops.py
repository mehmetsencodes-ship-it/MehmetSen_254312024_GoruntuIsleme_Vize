"""Klasik görüntü işleme operatörleri: yoğunluk dönüşümü, filtreleme,
segmentasyon, morfoloji ve temas eden nesnelerin ayrıştırılması."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


# --------------------------------------------------------------------------
# Yoğunluk dönüşümleri
# --------------------------------------------------------------------------
def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    img = (img - img.min()) / max(img.max() - img.min(), 1e-9)
    return (img * 255).round().astype(np.uint8)


def contrast_stretch(img, p_low=1.0, p_high=99.5):
    lo, hi = np.percentile(img, [p_low, p_high])
    out = (img.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def gamma_transform(img, gamma=0.5, p_high=99.5):
    hi = max(np.percentile(img, p_high), 1.0)
    out = np.clip(img.astype(np.float32) / hi, 0, 1) ** gamma
    return (out * 255).round().astype(np.uint8)


def log_transform(img, p_high=99.5):
    hi = max(np.percentile(img, p_high), 1.0)
    out = np.log1p(np.minimum(img.astype(np.float32), hi)) / np.log1p(hi)
    return (out * 255).round().astype(np.uint8)


def hist_equalize(img):
    return cv2.equalizeHist(img)


def clahe(img, clip_limit=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(img)


def hist_match(img, reference):
    out = exposure.match_histograms(img, reference)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def apply_intensity(img, cfg, reference=None):
    m = cfg.get("method", "none")
    if m == "none":
        return img
    if m == "stretch":
        return contrast_stretch(img, cfg.get("p_low", 1.0), cfg.get("p_high", 99.5))
    if m == "gamma":
        return gamma_transform(img, cfg.get("gamma", 0.5), cfg.get("p_high", 99.5))
    if m == "log":
        return log_transform(img, cfg.get("p_high", 99.5))
    if m == "hist_eq":
        return hist_equalize(img)
    if m == "clahe":
        return clahe(img, cfg.get("clip_limit", 2.0), cfg.get("tile", 8))
    if m == "hist_match":
        if reference is None:
            raise ValueError("hist_match için geliştirme kümesinden referans görüntü gerekir.")
        return hist_match(img, reference)
    raise ValueError(f"Bilinmeyen yoğunluk yöntemi: {m}")


# --------------------------------------------------------------------------
# Uzamsal filtreler
# --------------------------------------------------------------------------
SHARPEN_KERNELS = {
    "laplacian4": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    "laplacian8": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32),
}


def apply_filter(img, cfg):
    m = cfg.get("method", "none")
    k = cfg.get("ksize", 3)
    if m == "none":
        return img
    if m == "gaussian":
        return cv2.GaussianBlur(img, (k, k), cfg.get("sigma", 0))
    if m == "median":
        return cv2.medianBlur(img, k)
    if m == "mean":
        return cv2.blur(img, (k, k))
    if m == "bilateral":
        return cv2.bilateralFilter(img, k, cfg.get("sigma_color", 25), cfg.get("sigma_space", 5))
    if m == "sharpen":
        kernel = SHARPEN_KERNELS[cfg.get("kernel", "laplacian4")]
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    if m == "unsharp":
        blur = cv2.GaussianBlur(img, (k, k), 0).astype(np.float32)
        amount = cfg.get("amount", 1.0)
        out = img.astype(np.float32) + amount * (img.astype(np.float32) - blur)
        return np.clip(out, 0, 255).round().astype(np.uint8)
    raise ValueError(f"Bilinmeyen filtre: {m}")


# --------------------------------------------------------------------------
# Segmentasyon (ikili maske üretimi)
# --------------------------------------------------------------------------
def segment(img, cfg):
    m = cfg["method"]
    if m == "fixed":
        return img > cfg["threshold"]
    if m == "otsu":
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img > t
    if m == "triangle":
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return img > t
    if m == "adaptive":
        out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    cfg.get("block_size", 101), cfg.get("C", -2))
        mask = out > 0
        # Tamamen düz (0) arka planda adaptif eşik tanımsız kalır; global taban eşiği uygulanır.
        return mask & (img > cfg.get("floor", 0))
    if m == "kmeans":
        k = cfg.get("k", 3)
        data = img.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(centers.ravel())
        background = order[: cfg.get("n_background", 1)]
        return ~np.isin(labels.reshape(img.shape), background)
    raise ValueError(f"Bilinmeyen segmentasyon yöntemi: {m}")


# --------------------------------------------------------------------------
# Morfoloji
# --------------------------------------------------------------------------
def structuring_element(ksize, shape="ellipse"):
    shapes = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    return cv2.getStructuringElement(shapes[shape], (ksize, ksize))


def apply_morphology(mask, steps):
    m = mask.astype(np.uint8)
    for st in steps:
        op = st["op"]
        if op == "fill_holes":
            m = ndi.binary_fill_holes(m).astype(np.uint8)
            continue
        se = structuring_element(st.get("ksize", 3), st.get("shape", "ellipse"))
        it = st.get("iterations", 1)
        if op == "erode":
            m = cv2.erode(m, se, iterations=it)
        elif op == "dilate":
            m = cv2.dilate(m, se, iterations=it)
        elif op == "open":
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se, iterations=it)
        elif op == "close":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=it)
        else:
            raise ValueError(f"Bilinmeyen morfolojik işlem: {op}")
    return m.astype(bool)


# --------------------------------------------------------------------------
# Etiketleme ve temas eden hücrelerin ayrıştırılması
# --------------------------------------------------------------------------
def connected_components(mask, connectivity=8):
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return labels


def separate(mask, cfg):
    """Temas eden nesneleri distance transform + marker tabanlı watershed ile ayırır."""
    m = cfg.get("method", "none")
    if m == "none":
        return connected_components(mask), None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if cfg.get("dist_blur", 0):
        k = cfg["dist_blur"]
        dist = cv2.GaussianBlur(dist, (k, k), 0)
    cc = connected_components(mask)
    if m == "watershed_peaks":
        coords = peak_local_max(dist, min_distance=cfg.get("min_distance", 15),
                                threshold_abs=cfg.get("min_peak", 3), labels=cc,
                                exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    elif m == "watershed_hmax":
        peaks = h_maxima(dist, cfg.get("h", 2.0)) & mask
        markers = connected_components(peaks)
    else:
        raise ValueError(f"Bilinmeyen ayrıştırma yöntemi: {m}")
    # Marker içermeyen bağlı bileşenler kaybolmasın: kendilerine marker atanır.
    missing = np.setdiff1d(np.unique(cc[mask]), np.unique(cc[markers > 0]))
    nxt = markers.max() + 1
    for lab in missing:
        markers[cc == lab] = nxt
        nxt += 1
    labels = watershed(-dist, markers, mask=mask)
    return labels, dist


def filter_by_area(labels, min_area=0, max_area=None):
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    keep = areas >= min_area
    if max_area is not None:
        keep &= areas <= max_area
    kept = ids[keep]
    out = np.where(np.isin(labels, kept), labels, 0)
    return out, len(kept)
