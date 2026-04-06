"""
Fat Graft Predictor v3
======================
Three-signal hollowness fusion:
  1. MediaPipe z-depth + geometry (z-variance, convexity)
  2. OpenCV LAB luminance analysis
  3. MiDaS monocular depth estimation (local, no API)

Requirements:
    pip install opencv-python mediapipe numpy scipy
    pip install torch torchvision timm          # for MiDaS (optional but recommended)

Usage:
    python fat_graft_predictor_v3.py

DISCLAIMER:
    This tool is a clinical planning aid only.
    All volume recommendations MUST be reviewed and adjusted
    by the treating clinician before any procedure.
"""

import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import contextlib
import warnings
import io

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    from scipy.spatial import ConvexHull
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not found - convexity signal disabled.  pip install scipy")

# ── MiDaS (optional) ─────────────────────────────────────────────────────────
MIDAS_OK = False
midas_model = None
midas_transform = None

def _try_load_midas():
    global MIDAS_OK, midas_model, midas_transform
    try:
        import torch
        # Suppress warnings and hub loading noise
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        local_model_path = os.path.join("models", "midas_small.pt")
        
        # Capture hub stdout/stderr to suppress "Using cache" messages
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            # 1. Load the architecture (this may still need hub if no local cache)
            midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            
            # 2. Check for local weights to avoid download/hub cache dependency
            if os.path.exists(local_model_path):
                try:
                    state_dict = torch.load(local_model_path, map_location="cpu")
                    midas_model.load_state_dict(state_dict)
                except Exception:
                    pass # Fallback to standard hub behavior
                    
            midas_model.eval()
            
            # 3. Load transforms
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            midas_transform = transforms.small_transform
            
        MIDAS_OK = True
        print("[INFO] MiDaS Depth Signal: READY")
    except Exception as e:
        print(f"[WARN] MiDaS not available ({e}). Depth-map signal will be skipped.")
        MIDAS_OK = False

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    mp_solutions = mp.solutions
except Exception:
    raise ImportError("Install mediapipe:  pip install mediapipe")


# ------------------------------------------------------------------------─────
# LANDMARK INDEX GROUPS  (MediaPipe 468 pts)
# ------------------------------------------------------------------------─────
REGIONS = {
    "brows_L":       [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    "brows_R":       [66, 105, 63, 70, 46, 53, 52, 65, 55, 107],
    "forehead_L":    [338, 297, 332, 284, 251],
    "forehead_R":    [109, 67, 103, 54, 21],
    "temporals_L":   [234, 93, 132, 58, 172],
    "temporals_R":   [454, 323, 361, 288, 397],
    "sup_orbit_L":   [226, 113, 225, 224, 223, 222, 221, 189],
    "sup_orbit_R":   [446, 342, 445, 444, 443, 442, 441, 413],
    "infraorb_L":    [119, 100, 120, 121, 116, 123],
    "infraorb_R":    [348, 329, 349, 350, 345, 352],
    "soofs_L":       [36, 205, 206, 207, 187, 147],
    "soofs_R":       [266, 425, 426, 427, 411, 376],
    "buccal_L":      [213, 192, 214, 135, 172, 138],
    "buccal_R":      [433, 416, 434, 364, 397, 367],
    "nasolabial_L":  [216, 207, 187, 147, 213],
    "nasolabial_R":  [436, 427, 411, 376, 433],
    "chin":          [199, 175, 152, 148, 176, 149, 150, 136, 172],
    "mandibles_L":   [172, 136, 150, 149, 176, 148],
    "mandibles_R":   [397, 365, 379, 378, 400, 377],
    "marionette_L":  [202, 210, 169, 192, 213],
    "marionette_R":  [422, 430, 394, 416, 433],
    "pyriforms_L":   [129, 209, 49, 64, 98],
    "pyriforms_R":   [358, 429, 279, 294, 327],
    "nasal_dorsum":  [6, 197, 195, 5, 4],
    "nasal_tip":     [1, 2, 4, 5],
    "lips_upper":    [0, 267, 269, 270, 409, 306, 375, 321, 405, 314],
    "lips_lower":    [17, 84, 181, 91, 146, 61, 185, 40, 39, 37],
}

ARCH_MAP = {
    "brows_L": "T", "brows_R": "T", "forehead_L": "T", "forehead_R": "T",
    "temporals_L": "T", "temporals_R": "T", "sup_orbit_L": "T", "sup_orbit_R": "T",
    "infraorb_L": "M", "infraorb_R": "M", "soofs_L": "M", "soofs_R": "M",
    "buccal_L": "M", "buccal_R": "M", "nasolabial_L": "M", "nasolabial_R": "M",
    "chin": "J", "mandibles_L": "J", "mandibles_R": "J",
    "marionette_L": "J", "marionette_R": "J", "pyriforms_L": "J", "pyriforms_R": "J",
    "nasal_dorsum": "Other", "nasal_tip": "Other",
    "lips_upper": "Other", "lips_lower": "Other",
}

LABELS = {
    "brows_L": "Brows (L)",            "brows_R": "Brows (R)",
    "forehead_L": "Forehead (L)",      "forehead_R": "Forehead (R)",
    "temporals_L": "Temporals (L)",    "temporals_R": "Temporals (R)",
    "sup_orbit_L": "Sup Orbit (L)",    "sup_orbit_R": "Sup Orbit (R)",
    "infraorb_L": "Infraorbital (L)",  "infraorb_R": "Infraorbital (R)",
    "soofs_L": "SOOFs/Cheeks (L)",     "soofs_R": "SOOFs/Cheeks (R)",
    "buccal_L": "Buccal Fat (L)",      "buccal_R": "Buccal Fat (R)",
    "nasolabial_L": "Nasolabial (L)",  "nasolabial_R": "Nasolabial (R)",
    "chin": "Chin (midline)",
    "mandibles_L": "Mandibles (L)",    "mandibles_R": "Mandibles (R)",
    "marionette_L": "Marionette (L)",  "marionette_R": "Marionette (R)",
    "pyriforms_L": "Pyriforms (L)",    "pyriforms_R": "Pyriforms (R)",
    "nasal_dorsum": "Nasal Dorsum",    "nasal_tip": "Nasal Tip",
    "lips_upper": "Lips Upper",        "lips_lower": "Lips Lower",
}

ARCH_MF_PER_REGION      = {"T": 0.5, "M": 0.8, "J": 0.9, "Other": 0.5}
ARCH_NF_DEEP_PER_REGION = {"T": 0.0, "M": 0.4, "J": 0.4, "Other": 0.0}

NATURALLY_DEEP = {
    "nasal_tip", "nasal_dorsum",
    "sup_orbit_L", "sup_orbit_R",
    "lips_upper", "lips_lower",
}

# ── JMT INJECTION POINTS ─────────────────────────────────────────────────────
# Each JMT point maps to one or more anatomic regions for signal analysis
JMT_REGIONS = [
    {"key": "T1_brow",              "label": "T1 Brow",                         "arch": "T"},
    {"key": "T2_crest",             "label": "T2 Crest",                        "arch": "T"},
    {"key": "T3_hollowing",         "label": "T3 Hollowing",                    "arch": "T"},
    {"key": "M1_arcus_medial",      "label": "M1 Arcus marginalis 1/3 medial", "arch": "M"},
    {"key": "M2_arcus_median",      "label": "M2 Arcus marginalis 1/3 median", "arch": "M"},
    {"key": "M3_arcus_lateral",     "label": "M3 Arcus marginalis 1/3 lateral", "arch": "M"},
    {"key": "M4_zygomatic_arch",    "label": "M4 Zygomatic arch",              "arch": "M"},
    {"key": "M5_zygomatic_ligament","label": "M5 Zygomatic ligament",          "arch": "M"},
    {"key": "J1_chin_vertical",     "label": "J1 Chin vertical",               "arch": "J"},
    {"key": "J2_chin_horizontal",   "label": "J2 Chin horizontal",             "arch": "J"},
    {"key": "J3_melomental",        "label": "J3 Melomental area",             "arch": "J"},
    {"key": "J4_border",            "label": "J4 Border",                      "arch": "J"},
    {"key": "J5_angle",             "label": "J5 Angle",                       "arch": "J"},
    {"key": "AN_nose",              "label": "AN Nose",                        "arch": "Other"},
    {"key": "AL1_upper_lip",        "label": "AL1 Upper Lip",                  "arch": "Other"},
    {"key": "AL2_lower_lip",        "label": "AL2 Lower Lip",                  "arch": "Other"},
]

# Map each JMT injection point to underlying anatomic regions for hollow analysis
JMT_TO_ANATOMIC = {
    "T1_brow":              ["brows_L", "brows_R"],
    "T2_crest":             ["forehead_L", "forehead_R"],
    "T3_hollowing":         ["temporals_L", "temporals_R"],
    "M1_arcus_medial":      ["infraorb_L", "infraorb_R"],
    "M2_arcus_median":      ["infraorb_L", "infraorb_R", "soofs_L", "soofs_R"],
    "M3_arcus_lateral":     ["soofs_L", "soofs_R"],
    "M4_zygomatic_arch":    ["soofs_L", "soofs_R", "buccal_L", "buccal_R"],
    "M5_zygomatic_ligament":["nasolabial_L", "nasolabial_R"],
    "J1_chin_vertical":     ["chin"],
    "J2_chin_horizontal":   ["chin", "mandibles_L", "mandibles_R"],
    "J3_melomental":        ["marionette_L", "marionette_R"],
    "J4_border":            ["mandibles_L", "mandibles_R"],
    "J5_angle":             ["mandibles_L", "mandibles_R"],
    "AN_nose":              ["nasal_dorsum", "nasal_tip"],
    "AL1_upper_lip":        ["lips_upper"],
    "AL2_lower_lip":        ["lips_lower"],
}

# ── ANATOMIC SELECTABLE REGIONS (matches HTML UI) ────────────────────────────
ANATOMIC_SELECTABLE = [
    {"key": "brows",        "label": "Brows (L/R)",                 "paired": True,  "arch": "T",
     "regions_L": ["brows_L"],       "regions_R": ["brows_R"]},
    {"key": "forehead",     "label": "Forehead (L/R)",              "paired": True,  "arch": "T",
     "regions_L": ["forehead_L"],    "regions_R": ["forehead_R"]},
    {"key": "temporals",    "label": "Temporals (L/R)",             "paired": True,  "arch": "T",
     "regions_L": ["temporals_L"],   "regions_R": ["temporals_R"]},
    {"key": "sup_orbit_sulcus", "label": "Superior Orbit Sulcus (L/R)", "paired": True, "arch": "T",
     "regions_L": ["sup_orbit_L"],   "regions_R": ["sup_orbit_R"]},
    {"key": "infraorbital", "label": "Infraorbital (L/R)",          "paired": True,  "arch": "M",
     "regions_L": ["infraorb_L"],    "regions_R": ["infraorb_R"]},
    {"key": "soof_cheek",   "label": "SOOFs / Cheeks (L/R)",        "paired": True,  "arch": "M",
     "regions_L": ["soofs_L"],       "regions_R": ["soofs_R"]},
    {"key": "buccal_fat",   "label": "Buccal Fat (L/R)",            "paired": True,  "arch": "M",
     "regions_L": ["buccal_L"],      "regions_R": ["buccal_R"]},
    {"key": "nl_fold",      "label": "Nasolabial Folds (L/R)",      "paired": True,  "arch": "M",
     "regions_L": ["nasolabial_L"],  "regions_R": ["nasolabial_R"]},
    {"key": "marionette",   "label": "Marionette Lines (L/R)",      "paired": True,  "arch": "J",
     "regions_L": ["marionette_L"],  "regions_R": ["marionette_R"]},
    {"key": "mandible",     "label": "Mandibles (L/R)",             "paired": True,  "arch": "J",
     "regions_L": ["mandibles_L"],   "regions_R": ["mandibles_R"]},
    {"key": "chin",         "label": "Chin (midline)",              "paired": False, "arch": "J",
     "regions_L": ["chin"],          "regions_R": []},
    {"key": "dorsum",       "label": "Nasal Dorsum (midline)",      "paired": False, "arch": "Other",
     "regions_L": ["nasal_dorsum"],  "regions_R": []},
    {"key": "nasal_tip",    "label": "Nasal Tip (midline)",         "paired": False, "arch": "Other",
     "regions_L": ["nasal_tip"],     "regions_R": []},
    {"key": "lips_upper",   "label": "Lips \u2013 Upper (midline)",      "paired": False, "arch": "Other",
     "regions_L": ["lips_upper"],    "regions_R": []},
    {"key": "lips_lower",   "label": "Lips \u2013 Lower (midline)",      "paired": False, "arch": "Other",
     "regions_L": ["lips_lower"],    "regions_R": []},
    {"key": "pyriforms",    "label": "Pyriforms (L/R)",             "paired": True,  "arch": "J",
     "regions_L": ["pyriforms_L"],   "regions_R": ["pyriforms_R"]},
]

STABLE_PLANE_IDX = [
    36, 205, 207, 266, 425, 427,
    21, 54, 103, 67, 109,
    251, 284, 332, 297, 338,
    93, 132, 323, 361,
]

# cc recommendation table: (low, high, cc_string, severity_label)
HOLLOW_TO_CC_TABLE = [
    (0.00, 0.15, "0.0",    "None"),
    (0.15, 0.35, "+0.5",  "Minimal"),
    (0.35, 0.55, "+1.0",  "Moderate"),
    (0.55, 0.75, "+1.5",  "Significant"),
    (0.75, 1.01, "+2.0",  "Severe"),
]


# ------------------------------------------------------------------------─────
# SIGNAL 1 - Enhanced MediaPipe (z-depth + z-variance + convexity)
# ------------------------------------------------------------------------─────
def get_face_refs(landmarks):
    plane_z       = np.array([landmarks[i].z for i in STABLE_PLANE_IDX if i < len(landmarks)])
    all_z         = np.array([lm.z for lm in landmarks])
    face_median_z = float(np.median(plane_z))
    z_iqr         = float(np.percentile(all_z, 75) - np.percentile(all_z, 25))
    z_iqr         = max(z_iqr, 0.008)

    lx = landmarks[234].x
    rx = landmarks[454].x
    nx = landmarks[1].x
    if rx > lx:
        frac      = (nx - lx) / (rx - lx)
        yaw_score = float(np.clip(abs(frac - 0.5) * 2.5, 0.0, 1.0))
    else:
        yaw_score = 0.5

    return face_median_z, z_iqr, yaw_score


def detect_hollowness_v3(landmarks, indices, face_median_z, z_iqr,
                          region_name="", h=1, w=1):
    """
    Three-component MediaPipe hollowness:
      - depth score   (z below face plane)
      - flatness score (low z-variance = collapsed region)
      - concavity score (polygon area vs convex hull)
    """
    valid = [i for i in indices if i < len(landmarks)]
    if not valid:
        return 0.0, 0.0, 0.0, 0.0

    pts_3d = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z]
                        for i in valid])

    # -- depth score --
    avg_z     = float(np.mean(pts_3d[:, 2]))
    below_ref = face_median_z - avg_z
    threshold = 0.90 if region_name in NATURALLY_DEEP else 0.55
    depth_score = float(np.clip(2.0 * below_ref / z_iqr - threshold, 0.0, 1.0))

    # -- flatness score --
    z_std          = float(np.std(pts_3d[:, 2]))
    flatness_score = float(np.clip(1.0 - (z_std / (z_iqr * 0.4)), 0.0, 1.0))

    # -- concavity score --
    concavity_score = 0.0
    if SCIPY_OK and len(pts_3d) >= 4:
        pts_2d = pts_3d[:, :2].copy()
        pts_2d[:, 0] *= w
        pts_2d[:, 1] *= h
        try:
            hull      = ConvexHull(pts_2d)
            hull_area = hull.volume  # volume == area in 2-D
            x_, y_    = pts_2d[:, 0], pts_2d[:, 1]
            poly_area = 0.5 * abs(
                np.dot(x_, np.roll(y_, 1)) - np.dot(y_, np.roll(x_, 1))
            )
            convexity       = poly_area / (hull_area + 1e-6)
            concavity_score = float(np.clip(1.0 - convexity, 0.0, 1.0))
        except Exception:
            concavity_score = 0.0

    combined = (0.55 * depth_score +
                0.25 * flatness_score +
                0.20 * concavity_score)

    return (round(combined, 2),
            round(depth_score, 2),
            round(flatness_score, 2),
            round(concavity_score, 2))


# ------------------------------------------------------------------------─────
# SIGNAL 2 - OpenCV LAB luminance
# ------------------------------------------------------------------------─────
def build_face_hull_mask(landmarks, h, w):
    all_pts  = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks],
                         dtype=np.int32)
    mask     = np.zeros((h, w), dtype=np.uint8)
    hull     = cv2.convexHull(all_pts)
    cv2.fillPoly(mask, [hull], 255)
    return mask


def get_luminance_scores(frame_bgr, landmarks, h, w):
    """
    Returns {region: luminance_hollow_score 0-1}
    Uses LAB L-channel: darker region relative to face median = more hollow.
    """
    lab          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L            = lab[:, :, 0].astype(np.float32)
    face_mask    = build_face_hull_mask(landmarks, h, w)
    face_L_vals  = L[face_mask == 255]

    if len(face_L_vals) == 0:
        return {r: 0.0 for r in REGIONS}

    face_median_L = float(np.median(face_L_vals))
    face_iqr_L    = float(np.percentile(face_L_vals, 75) -
                          np.percentile(face_L_vals, 25))
    face_iqr_L    = max(face_iqr_L, 5.0)

    scores = {}
    for region, indices in REGIONS.items():
        valid = [i for i in indices if i < len(landmarks)]
        pts   = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                           for i in valid], dtype=np.int32)
        if len(pts) < 3:
            scores[region] = 0.0
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        region_L = L[mask == 255]

        if len(region_L) == 0:
            scores[region] = 0.0
            continue

        region_mean_L  = float(np.mean(region_L))
        darkness_ratio = (face_median_L - region_mean_L) / face_iqr_L
        scores[region] = round(float(np.clip(darkness_ratio * 0.8, 0.0, 1.0)), 2)

    return scores


# ------------------------------------------------------------------------─────
# SIGNAL 3 - MiDaS monocular depth
# ------------------------------------------------------------------------─────
def get_midas_scores(frame_rgb, landmarks, h, w):
    """
    Returns {region: midas_hollow_score 0-1}
    MiDaS inverse-depth: lower value = farther from camera = more hollow.
    """
    if not MIDAS_OK:
        return {r: 0.0 for r in REGIONS}

    import torch
    inp    = midas_transform(frame_rgb)
    with torch.no_grad():
        depth = midas_model(inp)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().numpy()

    d_min, d_max = depth.min(), depth.max()
    depth_norm   = (depth - d_min) / (d_max - d_min + 1e-6)

    face_mask      = build_face_hull_mask(landmarks, h, w)
    face_depth_vals = depth_norm[face_mask == 255]

    if len(face_depth_vals) == 0:
        return {r: 0.0 for r in REGIONS}

    face_median = float(np.median(face_depth_vals))
    face_iqr    = float(np.percentile(face_depth_vals, 75) -
                        np.percentile(face_depth_vals, 25))
    face_iqr    = max(face_iqr, 0.02)

    scores = {}
    for region, indices in REGIONS.items():
        valid = [i for i in indices if i < len(landmarks)]
        pts   = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                           for i in valid], dtype=np.int32)
        if len(pts) < 3:
            scores[region] = 0.0
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        region_depth = depth_norm[mask == 255]

        if len(region_depth) == 0:
            scores[region] = 0.0
            continue

        region_mean  = float(np.mean(region_depth))
        hollow       = np.clip((face_median - region_mean) / face_iqr - 0.3, 0.0, 1.0)
        scores[region] = round(float(hollow), 2)

    return scores


# ------------------------------------------------------------------------─────
# FUSION
# ------------------------------------------------------------------------─────
def fuse_signals(mp_score, lum_score, midas_score, yaw_confidence,
                  lum_available=True, midas_available=True):
    """
    Combines three hollow signals with spread-aware weighting.
    Returns (fused_score, signal_confidence_label).

    When a signal is unavailable (not computed) it is excluded gracefully.
    """
    active = [mp_score]
    if lum_available:
        active.append(lum_score)
    if midas_available:
        active.append(midas_score)

    spread = max(active) - min(active)

    if len(active) == 1:
        fused = mp_score
        conf_label = "mp-only"
    elif spread < 0.15:
        # All signals agree → equal-ish blend
        if midas_available:
            fused = 0.40 * mp_score + 0.30 * lum_score + 0.30 * midas_score
        else:
            fused = 0.60 * mp_score + 0.40 * lum_score
        conf_label = "high"
    elif spread < 0.30:
        # Mild disagreement → trust geometry more
        if midas_available:
            fused = 0.55 * mp_score + 0.25 * lum_score + 0.20 * midas_score
        else:
            fused = 0.70 * mp_score + 0.30 * lum_score
        conf_label = "medium"
    else:
        # Signals diverge → fall back to z-depth only
        fused      = mp_score
        conf_label = "low (signals diverged)"

    fused = float(np.clip(fused * yaw_confidence, 0.0, 1.0))
    return round(fused, 2), conf_label


def hollow_to_recommendation(score):
    for lo, hi, cc, label in HOLLOW_TO_CC_TABLE:
        if lo <= score < hi:
            return cc, label
    return None, "None"


# ------------------------------------------------------------------------─────
# VOLUME CALCULATION  (unchanged formula, new hollow scoring)
# ------------------------------------------------------------------------─────
def calc_fat_volumes(age, landmarks, frame_bgr, h, w):
    total_mf           = round((2 / 3) * age, 1)
    total_nf           = round((1 / 3) * age, 1)
    total_nf_epidermal = round(total_nf / 3, 1)
    total_nf_deep      = round(total_nf - total_nf_epidermal, 1)

    sum_mf_w = sum(ARCH_MF_PER_REGION[ARCH_MAP[r]]      for r in REGIONS)
    sum_nf_w = sum(ARCH_NF_DEEP_PER_REGION[ARCH_MAP[r]] for r in REGIONS)

    face_median_z, z_iqr, yaw = get_face_refs(landmarks)
    yaw_confidence = float(np.clip(1.0 - yaw * 1.8, 0.05, 1.0))

    # ── Compute the three signals ────────────────────────────────────────────
    print("[INFO] Computing luminance signal...")
    lum_scores = get_luminance_scores(frame_bgr, landmarks, h, w)

    print("[INFO] Computing MiDaS depth signal...")
    frame_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    midas_scores = get_midas_scores(frame_rgb, landmarks, h, w)

    results = {}
    for region, indices in REGIONS.items():
        arch  = ARCH_MAP[region]

        # Signal 1: enhanced MediaPipe
        mp_hollow, dep_s, flat_s, conc_s = detect_hollowness_v3(
            landmarks, indices, face_median_z, z_iqr, region, h, w
        )

        # Signal 2 & 3
        lum_s   = lum_scores.get(region, 0.0)
        midas_s = midas_scores.get(region, 0.0)

        # Fuse
        fused_hollow, sig_conf = fuse_signals(
            mp_hollow, lum_s, midas_s, yaw_confidence,
            lum_available=True,
            midas_available=MIDAS_OK,
        )

        # cc recommendation
        cc_add, severity = hollow_to_recommendation(fused_hollow)

        # Base volumes (age-scaled, hollowness doesn't change base cc)
        mf = round(ARCH_MF_PER_REGION[arch]      / sum_mf_w * total_mf, 1)
        nf = (round(ARCH_NF_DEEP_PER_REGION[arch] / sum_nf_w * total_nf_deep, 1)
              if sum_nf_w > 0 else 0.0)

        results[region] = {
            "label":        LABELS[region],
            "arch":         arch,
            "mf":           mf,
            "nf":           nf,
            "hollow":       fused_hollow,
            "hollow_mp":    mp_hollow,
            "hollow_lum":   lum_s,
            "hollow_midas": midas_s,
            "sig_conf":     sig_conf,
            "cc_add":       cc_add,
            "severity":     severity,
            # sub-signals for transparency
            "depth_s":      dep_s,
            "flatness_s":   flat_s,
            "concavity_s":  conc_s,
        }

    return {
        "regions":            results,
        "total_mf":           total_mf,
        "total_nf":           total_nf,
        "total_nf_deep":      total_nf_deep,
        "total_nf_epidermal": total_nf_epidermal,
        "yaw":                round(yaw, 2),
        "confidence":         round(yaw_confidence, 2),
        "midas_used":         MIDAS_OK,
    }


# ── ANATOMIC VOLUME CALCULATION (HTML-matching formula) ──────────────────────
def calc_anatomic_volumes(age, fat_data, selected_anat=None):
    """
    Calculate volumes for Anatomic regions mode using the HTML formula.
    Equal distribution per slot per arch among selected regions.
    Paired regions count as 2 slots (Left + Right).
    """
    total_mf = round((2 / 3) * age, 1)
    total_nf = round((1 / 3) * age, 1)
    total_nf_epidermal = round(total_nf / 3, 1)
    total_nf_deep = round(total_nf - total_nf_epidermal, 1)

    # MF per arch: 1/3 each for J, M, remaining; remaining splits 2/3 T + 1/3 Other
    mf_j = total_mf / 3
    mf_m = total_mf / 3
    mf_remaining = total_mf / 3
    mf_t = (2 / 3) * mf_remaining
    mf_other = (1 / 3) * mf_remaining
    MF_ARCH = {"J": mf_j, "M": mf_m, "T": mf_t, "Other": mf_other}

    # NF deep: 2/3 of NF total, split among J+M
    nf_jm = (2 / 3) * total_nf
    nf_epi = (1 / 3) * total_nf

    if selected_anat is None:
        selected_anat = {r["key"] for r in ANATOMIC_SELECTABLE}

    # Count selected slots per arch (paired=2, unpaired=1)
    arch_slots = {"J": 0, "M": 0, "T": 0, "Other": 0}
    for r in ANATOMIC_SELECTABLE:
        if r["key"] in selected_anat:
            arch_slots[r["arch"]] += 2 if r["paired"] else 1

    # Count selected J+M slots for NF deep distribution
    selected_jm_slots = 0
    for r in ANATOMIC_SELECTABLE:
        if r["key"] in selected_anat and r["arch"] in ("J", "M"):
            selected_jm_slots += 2 if r["paired"] else 1
    nf_per_jm_slot = nf_jm / selected_jm_slots if selected_jm_slots > 0 else 0.0

    # Build results - each paired region creates Left + Right entries
    anat_results = []
    for r in ANATOMIC_SELECTABLE:
        key = r["key"]
        arch = r["arch"]
        is_selected = key in selected_anat

        # Get hollow scores from underlying landmark regions
        all_regions = r["regions_L"] + r["regions_R"]
        hollows = []
        for rk in all_regions:
            if rk in fat_data["regions"]:
                hollows.append(fat_data["regions"][rk]["hollow"])
        avg_hollow = round(float(np.mean(hollows)), 2) if hollows else 0.0
        cc_add, severity = hollow_to_recommendation(avg_hollow)

        if is_selected:
            slots = arch_slots[arch]
            per_slot_mf = MF_ARCH[arch] / slots if slots > 0 else 0.0
            per_slot_nf = nf_per_jm_slot if arch in ("J", "M") else 0.0
        else:
            per_slot_mf = 0.0
            per_slot_nf = 0.0

        if r["paired"]:
            # Left entry
            anat_results.append({
                "area":     r["label"],
                "side":     "Left",
                "arch":     arch,
                "mf":       round(per_slot_mf, 1),
                "nf":       round(per_slot_nf, 1),
                "hollow":   avg_hollow,
                "severity": severity,
                "cc_add":   cc_add,
                "selected": is_selected,
            })
            # Right entry
            anat_results.append({
                "area":     r["label"],
                "side":     "Right",
                "arch":     arch,
                "mf":       round(per_slot_mf, 1),
                "nf":       round(per_slot_nf, 1),
                "hollow":   avg_hollow,
                "severity": severity,
                "cc_add":   cc_add,
                "selected": is_selected,
            })
        else:
            anat_results.append({
                "area":     r["label"],
                "side":     "Midline",
                "arch":     arch,
                "mf":       round(per_slot_mf, 1),
                "nf":       round(per_slot_nf, 1),
                "hollow":   avg_hollow,
                "severity": severity,
                "cc_add":   cc_add,
                "selected": is_selected,
            })

    return {
        "rows":               anat_results,
        "total_mf":           total_mf,
        "total_nf":           total_nf,
        "total_nf_deep":      total_nf_deep,
        "total_nf_epidermal": round(nf_epi, 1),
        "mf_arch":            MF_ARCH,
        "nf_jm":              round(nf_jm, 1),
        "yaw":                fat_data.get("yaw", 0.0),
        "confidence":         fat_data.get("confidence", 1.0),
        "midas_used":         fat_data.get("midas_used", False),
    }


# ── JMT VOLUME CALCULATION ───────────────────────────────────────────────────
def calc_jmt_volumes(age, fat_data, selected_jmt=None):
    """
    Calculate volumes for JMT injection points mode.
    Uses hte same 2/3 formula and distributes equally per slot per arch.
    Hollow scores are averaged from mapped anatomic regions.
    
    selected_jmt: set of JMT keys that are selected (None = all selected)
    """
    total_mf = round((2 / 3) * age, 1)
    total_nf = round((1 / 3) * age, 1)
    total_nf_epidermal = round(total_nf / 3, 1)
    total_nf_deep = round(total_nf - total_nf_epidermal, 1)

    # MF per arch: 1/3 each for J, M, remaining; remaining splits 2/3 T + 1/3 Other
    mf_j = total_mf / 3
    mf_m = total_mf / 3
    mf_remaining = total_mf / 3
    mf_t = (2 / 3) * mf_remaining
    mf_other = (1 / 3) * mf_remaining
    MF_ARCH = {"J": mf_j, "M": mf_m, "T": mf_t, "Other": mf_other}

    # NF deep only for J+M
    nf_jm = (2 / 3) * total_nf

    if selected_jmt is None:
        selected_jmt = {r["key"] for r in JMT_REGIONS}

    # Count selected slots per arch (all JMT points are midline = 1 slot each)
    arch_slots = {"J": 0, "M": 0, "T": 0, "Other": 0}
    for r in JMT_REGIONS:
        if r["key"] in selected_jmt:
            arch_slots[r["arch"]] += 1

    # Count selected J+M slots for NF distribution
    selected_jm_slots = 0
    for r in JMT_REGIONS:
        if r["key"] in selected_jmt and r["arch"] in ("J", "M"):
            selected_jm_slots += 1

    nf_per_jm_slot = nf_jm / selected_jm_slots if selected_jm_slots > 0 else 0.0

    # Build JMT results
    jmt_results = {}
    for r in JMT_REGIONS:
        key = r["key"]
        arch = r["arch"]
        is_selected = key in selected_jmt

        # Get hollow score by averaging mapped anatomic regions
        mapped = JMT_TO_ANATOMIC.get(key, [])
        hollows = []
        hollow_mp_vals = []
        hollow_lum_vals = []
        hollow_midas_vals = []
        for anat_key in mapped:
            if anat_key in fat_data["regions"]:
                v = fat_data["regions"][anat_key]
                hollows.append(v["hollow"])
                hollow_mp_vals.append(v["hollow_mp"])
                hollow_lum_vals.append(v["hollow_lum"])
                hollow_midas_vals.append(v["hollow_midas"])

        avg_hollow = round(float(np.mean(hollows)), 2) if hollows else 0.0
        avg_mp     = round(float(np.mean(hollow_mp_vals)), 2) if hollow_mp_vals else 0.0
        avg_lum    = round(float(np.mean(hollow_lum_vals)), 2) if hollow_lum_vals else 0.0
        avg_midas  = round(float(np.mean(hollow_midas_vals)), 2) if hollow_midas_vals else 0.0

        cc_add, severity = hollow_to_recommendation(avg_hollow)

        if is_selected:
            slots = arch_slots[arch]
            mf = round(MF_ARCH[arch] / slots, 1) if slots > 0 else 0.0
            nf = round(nf_per_jm_slot, 1) if arch in ("J", "M") else 0.0
        else:
            mf = 0.0
            nf = 0.0

        # Get best signal confidence from mapped regions
        sig_confs = []
        for anat_key in mapped:
            if anat_key in fat_data["regions"]:
                sig_confs.append(fat_data["regions"][anat_key]["sig_conf"])
        sig_conf = sig_confs[0] if sig_confs else "mp-only"

        jmt_results[key] = {
            "label":        r["label"],
            "arch":         arch,
            "mf":           mf,
            "nf":           nf,
            "hollow":       avg_hollow,
            "hollow_mp":    avg_mp,
            "hollow_lum":   avg_lum,
            "hollow_midas": avg_midas,
            "sig_conf":     sig_conf,
            "cc_add":       cc_add,
            "severity":     severity,
            "side":         "Midline",
        }

    return {
        "regions":            jmt_results,
        "total_mf":           total_mf,
        "total_nf":           total_nf,
        "total_nf_deep":      total_nf_deep,
        "total_nf_epidermal": total_nf_epidermal,
        "mf_arch":            MF_ARCH,
        "nf_jm":              round(nf_jm, 1),
        "yaw":                fat_data.get("yaw", 0.0),
        "confidence":         fat_data.get("confidence", 1.0),
        "midas_used":         fat_data.get("midas_used", False),
    }


# ------------------------------------------------------------------------─────
# -----------------------------------------------------------------------------
# COLOR HELPERS
# -----------------------------------------------------------------------------
# Severity badge colours (BGR)
SEVERITY_BGR = {
    "None":        (100, 180, 100),
    "Minimal":     (100, 220, 220),
    "Moderate":    (60,  190, 255),
    "Significant": (40,  120, 255),
    "Severe":      (60,   60, 220),
}

_D_LUT = np.zeros((101, 3), dtype=np.uint8)
for _i in range(101):
    _D_LUT[_i] = cv2.cvtColor(
        np.uint8([[[int(_i / 100 * 120), 180, 210]]]),
        cv2.COLOR_HSV2BGR)[0][0]
_DEPTH_STRIP = np.zeros((12, 100, 3), dtype=np.uint8)
for _i in range(100):
    _DEPTH_STRIP[:, _i] = _D_LUT[min(int(_i / 99 * 100), 100)]


def hollow_color(severity):
    return SEVERITY_BGR.get(severity, (150, 150, 150))


# -----------------------------------------------------------------------------
# DRAW OVERLAY ON FACE IMAGE
# -----------------------------------------------------------------------------
def draw_overlay(frame, landmarks, fat_data, h, w, face_mesh_module):
    # Mesh contour lines
    for conn in face_mesh_module.FACEMESH_CONTOURS:
        a, b = conn
        if a < len(landmarks) and b < len(landmarks):
            x1 = int(landmarks[a].x * w); y1 = int(landmarks[a].y * h)
            x2 = int(landmarks[b].x * w); y2 = int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

    overlay = frame.copy()
    deep_idx_set = set()
    for rname in NATURALLY_DEEP:
        deep_idx_set.update(REGIONS.get(rname, []))

    for region, indices in REGIONS.items():
        v      = fat_data["regions"][region]
        mf     = v["mf"]
        hollow = v["hollow"]
        severity = v["severity"]
        if mf == 0:
            continue

        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
               for i in indices if i < len(landmarks)]
        if len(pts) < 3:
            continue

        pts_arr   = np.array(pts, dtype=np.int32)
        color     = hollow_color(severity)
        
        if severity != "None":
            cv2.fillPoly(overlay, [pts_arr], color)
            thickness = 2 if hollow > 0.45 else 1
            cv2.polylines(frame, [pts_arr], True, color, thickness)
        else:
            # For None, just draw a faint outline
            cv2.polylines(frame, [pts_arr], True, (100, 150, 100), 1)

        cx  = int(np.mean([p[0] for p in pts]))
        cy  = int(np.mean([p[1] for p in pts]))

        # Show fused hollow score and cc recommendation
        cc  = str(v["cc_add"]) if v["cc_add"] is not None else "0.0"
        if cc == "0.0" and severity != "None":
            cc = ""
        lbl = f"{mf:.1f}  {cc}"
        flag = "!" if hollow > 0.45 else ""
        lbl  = flag + lbl

        # Only label if not None to reduce clutter
        if severity != "None":
            cv2.putText(frame, lbl, (cx - 18, cy + 5),
                        cv2.FONT_HERSHEY_PLAIN, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (cx - 17, cy + 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.72, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    # Depth dots
    xs    = np.clip((np.array([lm.x for lm in landmarks]) * w).astype(int), 0, w - 1)
    ys    = np.clip((np.array([lm.y for lm in landmarks]) * h).astype(int), 0, h - 1)
    zs    = np.array([lm.z for lm in landmarks])
    idxs  = np.clip(((zs + 0.08) / 0.16 * 100).astype(int), 0, 100)
    dot_colors = _D_LUT[idxs]
    for li, (xi, yi, ci) in enumerate(zip(xs, ys, dot_colors)):
        r = 1 if li in deep_idx_set else 2
        cv2.circle(frame, (int(xi), int(yi)), r,
                   (int(ci[0]), int(ci[1]), int(ci[2])), -1)

    # Legend - fat heatmap
    lx, ly = 10, h - 90
    cv2.rectangle(frame, (lx - 4, ly - 22), (lx + 120, ly + 80), (10, 10, 10), -1)
    cv2.rectangle(frame, (lx - 4, ly - 22), (lx + 120, ly + 80), (60, 60, 60), 1)
    cv2.putText(frame, "Fat needed:", (lx, ly - 7),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (200, 200, 200), 1)
    
    y_offset = ly + 10
    for sev, col in [("Minimal", (100, 220, 220)), ("Moderate", (60, 190, 255)), 
                     ("Significant", (40, 120, 255)), ("Severe", (60, 60, 220))]:
        cv2.rectangle(frame, (lx, y_offset), (lx + 15, y_offset + 10), col, -1)
        cv2.putText(frame, sev, (lx + 25, y_offset + 9), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)
        y_offset += 15

    # Legend - depth
    dx = lx + 130
    cv2.rectangle(frame, (dx - 4, ly - 22), (dx + 157, ly + 30), (10, 10, 10), -1)
    cv2.rectangle(frame, (dx - 4, ly - 22), (dx + 157, ly + 30), (60, 60, 60), 1)
    cv2.putText(frame, "Depth:", (dx, ly - 7),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (200, 200, 200), 1)
    frame[ly + 2:ly + 14, dx:dx + 100] = _DEPTH_STRIP
    cv2.putText(frame, "sunken", (dx,      ly + 27), cv2.FONT_HERSHEY_PLAIN, 0.7, (80,  80,  255), 1)
    cv2.putText(frame, "raised", (dx + 72, ly + 27), cv2.FONT_HERSHEY_PLAIN, 0.7, (100, 255, 100), 1)

    # HUD totals
    mf_t = fat_data["total_mf"]; nf_t = fat_data["total_nf"]
    cv2.rectangle(frame, (w - 198, 4), (w - 4, 58), (10, 10, 10), -1)
    cv2.rectangle(frame, (w - 198, 4), (w - 4, 58), (70, 70, 70), 1)
    cv2.putText(frame, f"MF: {mf_t:.1f} cc", (w - 190, 26),
                cv2.FONT_HERSHEY_PLAIN, 1.1, (160, 230, 110), 1)
    cv2.putText(frame, f"NF: {nf_t:.1f} cc", (w - 190, 50),
                cv2.FONT_HERSHEY_PLAIN, 1.1, (110, 190, 255), 1)

    # Signals badge
    signals_used = ["MP", "LUM"]
    if fat_data.get("midas_used"):
        signals_used.append("MIDAS")
    sig_txt = "Signals: " + "+".join(signals_used)
    cv2.rectangle(frame, (w - 198, 62), (w - 4, 80), (10, 10, 10), -1)
    cv2.putText(frame, sig_txt, (w - 194, 76),
                cv2.FONT_HERSHEY_PLAIN, 0.75, (160, 160, 255), 1)

    # Yaw warning
    yaw  = fat_data.get("yaw", 0.0)
    conf = fat_data.get("confidence", 1.0)
    if yaw > 0.15:
        warn_col = (0, 50, 255) if yaw > 0.40 else (0, 165, 255)
        warn_msg = (f"! FACE NOT FRONTAL  ({int(conf*100)}% accuracy)"
                    if yaw > 0.40 else
                    f"~ Slight angle  ({int(conf*100)}% accuracy)")
        cv2.rectangle(frame, (6, 35), (len(warn_msg) * 8 + 12, 56), (10, 10, 10), -1)
        cv2.putText(frame, warn_msg, (10, 51),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, warn_col, 1)

# STATS PANEL (right-side)
# ------------------------------------------------------------------------─────
ARCH_NAMES_PANEL = {
    "T": "T-Arch  Upper Face",
    "M": "M-Arch  Midface",
    "J": "J-Arch  Jawline",
    "Other": "Other",
}
ARCH_BGR_PANEL = {
    "T": (220, 190, 70), "M": (90, 200, 90),
    "J": (60, 175, 255), "Other": (190, 120, 200),
}


def build_stats_panel(anat_data, age, cam_h):
    """Build a stats panel for Anatomic regions mode (HTML-matching format)."""
    PW    = 420
    panel = np.full((cam_h, PW, 3), 18, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, cam_h), (55, 55, 55), 1)

    y = 22
    cv2.putText(panel, f"ANATOMIC REGIONS  |  Age: {age}", (8, y),
                cv2.FONT_HERSHEY_PLAIN, 1.05, (230, 230, 230), 1)
    y += 14
    nf_d = anat_data.get("total_nf_deep",      anat_data["total_nf"])
    nf_e = anat_data.get("total_nf_epidermal", 0.0)
    cv2.putText(panel,
                f"MF {anat_data['total_mf']:.1f}cc  NFd {nf_d:.1f}+{nf_e:.1f}cc",
                (8, y), cv2.FONT_HERSHEY_PLAIN, 0.95, (160, 230, 110), 1)
    y += 13

    # MF by arch
    mfa = anat_data.get("mf_arch", {})
    if mfa:
        cv2.putText(panel,
                    f"J:{mfa.get('J',0):.1f} M:{mfa.get('M',0):.1f} T:{mfa.get('T',0):.1f} O:{mfa.get('Other',0):.1f}",
                    (8, y), cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 200, 200), 1)
        y += 12

    conf     = anat_data.get("confidence", 1.0)
    yaw      = anat_data.get("yaw", 0.0)
    conf_col = ((100, 255, 100) if conf > 0.75 else
                ((0, 210, 255)  if conf > 0.40 else (60, 60, 255)))
    cv2.putText(panel, f"Face accuracy: {int(conf*100)}%  Yaw:{yaw:.2f}",
                (8, y), cv2.FONT_HERSHEY_PLAIN, 0.75, conf_col, 1)
    y += 12
    cv2.line(panel, (4, y), (PW - 4, y), (55, 55, 55), 1)
    y += 8

    # Column header matching HTML: Area, Side, MF, NF Deep, Arch
    cv2.putText(panel, "  Area                 Side    MF   NF   Arch",
                (4, y), cv2.FONT_HERSHEY_PLAIN, 0.60, (75, 75, 75), 1)
    y += 10
    cv2.line(panel, (4, y), (PW - 4, y), (40, 40, 40), 1)
    y += 6

    arch_order = [
        {"key": "J",     "title": "J Arch - Jawline / Lower Face"},
        {"key": "M",     "title": "M Arch - Midface"},
        {"key": "T",     "title": "T Arch - Temples / Upper Face"},
        {"key": "Other", "title": "Other - Nose / Lips / Misc"},
    ]

    rows = anat_data.get("rows", [])
    for ai in arch_order:
        arch_rows = [r for r in rows if r["arch"] == ai["key"]]
        if not arch_rows or y > cam_h - 28:
            continue
        color = ARCH_BGR_PANEL[ai["key"]]
        cv2.putText(panel, ai["title"], (6, y),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1)
        y += 12

        # Show non-zero first, then zero
        nz = [r for r in arch_rows if r["mf"] > 0]
        zr = [r for r in arch_rows if r["mf"] == 0]
        for r in nz + zr:
            if y > cam_h - 28:
                break
            area_short = r["area"][:20]
            txt_col = color if r["mf"] > 0 else (75, 75, 75)
            row_txt = f"  {area_short:<20} {r['side']:<7} {r['mf']:>4.1f} {r['nf']:>4.1f}  {r['arch']}"
            cv2.putText(panel, row_txt, (4, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.60, txt_col, 1)
            y += 11
        cv2.line(panel, (4, y), (PW - 4, y), (40, 40, 40), 1)
        y += 6

    # Totals
    if y < cam_h - 50:
        cv2.putText(panel, f"Totals:  MF {anat_data['total_mf']:.1f} cc   NF {anat_data['total_nf']:.1f} cc",
                    (8, y + 5), cv2.FONT_HERSHEY_PLAIN, 0.75, (200, 200, 200), 1)
        y += 15
        cv2.putText(panel, f"NF Epidermal (micro-needling): {nf_e:.1f} cc",
                    (8, y + 2), cv2.FONT_HERSHEY_PLAIN, 0.70, (110, 190, 255), 1)

    # Footer
    cv2.putText(panel, "PLANNING AID ONLY - clinician review required",
                (8, cam_h - 18), cv2.FONT_HERSHEY_PLAIN, 0.72, (60, 60, 180), 1)
    cv2.putText(panel, "Fat Graft Predictor v3 - Anatomic",
                (8, cam_h - 6),  cv2.FONT_HERSHEY_PLAIN, 0.65, (60, 60, 60), 1)
    return panel


def build_jmt_stats_panel(jmt_data, age, cam_h):
    """Build a stats panel for JMT injection points mode."""
    PW    = 420
    panel = np.full((cam_h, PW, 3), 18, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, cam_h), (55, 55, 55), 1)

    y = 22
    cv2.putText(panel, f"JMT INJECTION POINTS  |  Age: {age}", (8, y),
                cv2.FONT_HERSHEY_PLAIN, 1.05, (230, 230, 230), 1)
    y += 14
    nf_d = jmt_data.get("total_nf_deep",      jmt_data["total_nf"])
    nf_e = jmt_data.get("total_nf_epidermal", 0.0)
    cv2.putText(panel,
                f"MF {jmt_data['total_mf']:.1f}cc  NFd {nf_d:.1f}+{nf_e:.1f}cc",
                (8, y), cv2.FONT_HERSHEY_PLAIN, 0.95, (160, 230, 110), 1)
    y += 13

    # MF by arch
    mfa = jmt_data.get("mf_arch", {})
    if mfa:
        cv2.putText(panel,
                    f"J:{mfa.get('J',0):.1f} M:{mfa.get('M',0):.1f} T:{mfa.get('T',0):.1f} O:{mfa.get('Other',0):.1f}",
                    (8, y), cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 200, 200), 1)
        y += 12

    conf     = jmt_data.get("confidence", 1.0)
    yaw      = jmt_data.get("yaw", 0.0)
    conf_col = ((100, 255, 100) if conf > 0.75 else
                ((0, 210, 255)  if conf > 0.40 else (60, 60, 255)))
    cv2.putText(panel, f"Face accuracy: {int(conf*100)}%  Yaw:{yaw:.2f}",
                (8, y), cv2.FONT_HERSHEY_PLAIN, 0.75, conf_col, 1)
    y += 12
    cv2.line(panel, (4, y), (PW - 4, y), (55, 55, 55), 1)
    y += 8

    # Column header
    cv2.putText(panel, "  Point                          MF  NFd  Hollow  Severity",
                (4, y), cv2.FONT_HERSHEY_PLAIN, 0.58, (75, 75, 75), 1)
    y += 10
    cv2.line(panel, (4, y), (PW - 4, y), (40, 40, 40), 1)
    y += 6

    arch_names_jmt = {
        "J": "J Arch - Jawline / Lower Face",
        "M": "M Arch - Midface",
        "T": "T Arch - Temples / Upper Face",
        "Other": "Other - Nose / Lips / Misc",
    }

    for arch in ["T", "M", "J", "Other"]:
        if y > cam_h - 28:
            break
        color = ARCH_BGR_PANEL[arch]
        cv2.putText(panel, arch_names_jmt[arch], (6, y),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1)
        y += 12
        for key, v in jmt_data["regions"].items():
            if v["arch"] != arch or y > cam_h - 28:
                continue
            hc      = hollow_color(v["severity"])
            label   = v["label"][:28]
            cc_str  = str(v["cc_add"]) if v["cc_add"] else "0.0"
            sev_col = SEVERITY_BGR.get(v["severity"], (130, 130, 130))

            row_txt = f"  {label:<28} {v['mf']:>4.1f} {v['nf']:>4.1f} {v['hollow']:>6.2f}"
            cv2.putText(panel, row_txt, (4, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.62, hc, 1)
            # severity label
            cv2.putText(panel, v["severity"][:8], (PW - 60, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.60, sev_col, 1)

            # Hollow bar
            bar_w = int(v["hollow"] * 25)
            if bar_w > 0:
                cv2.rectangle(panel,
                              (PW - 95, y - 8), (PW - 95 + bar_w, y - 2), hc, -1)
            y += 11
        cv2.line(panel, (4, y), (PW - 4, y), (40, 40, 40), 1)
        y += 6

    # NF Epidermal note
    if y < cam_h - 40:
        cv2.putText(panel, f"NF Epidermal (micro-needling): {nf_e:.1f} cc",
                    (8, y + 5), cv2.FONT_HERSHEY_PLAIN, 0.70, (110, 190, 255), 1)

    # Footer
    cv2.putText(panel, "PLANNING AID ONLY - clinician review required",
                (8, cam_h - 18), cv2.FONT_HERSHEY_PLAIN, 0.72, (60, 60, 180), 1)
    cv2.putText(panel, "Fat Graft Predictor v3 - JMT Mode",
                (8, cam_h - 6),  cv2.FONT_HERSHEY_PLAIN, 0.65, (60, 60, 60), 1)
    return panel


# ------------------------------------------------------------------------─────
# CSV EXPORT
# ------------------------------------------------------------------------─────
def export_csv(data, age, image_path, mode="anatomic"):
    base     = os.path.splitext(image_path)[0]
    suffix   = "_jmt" if mode == "jmt" else ""
    csv_path = f"{base}_result{suffix}.csv"

    if mode == "anatomic":
        # Simplified CSV matching the HTML table: Area, Side, MF, NF Deep, Arch
        lines = ["Area,Side,MF (cc),NF Deep (cc),Arch"]
        rows = data.get("rows", [])
        arch_order = ["J", "M", "T", "Other"]
        arch_titles = {
            "J": "J Arch — Jawline / Lower Face",
            "M": "M Arch — Midface",
            "T": "T Arch — Temples / Upper Face",
            "Other": "Other — Nose / Lips / Misc",
        }
        for arch in arch_order:
            arch_rows = [r for r in rows if r["arch"] == arch]
            if not arch_rows:
                continue
            lines.append(f"{arch_titles[arch]},,,,{arch}")
            for r in arch_rows:
                lines.append(f"{r['area']},{r['side']},{r['mf']:.1f},{r['nf']:.1f},{r['arch']}")
        lines += [
            "",
            f"Totals,,{data['total_mf']:.1f},{data['total_nf']:.1f},",
            f"NF Epidermal (micro-needling),,,,{data['total_nf_epidermal']:.1f}",
        ]
    else:
        # JMT CSV
        lines = ["Area,Side,MF (cc),NF Deep (cc),Arch"]
        for region, v in data["regions"].items():
            side = v.get("side", "Midline")
            lines.append(
                f"{v['label']},{side},{v['mf']:.1f},{v['nf']:.1f},{v['arch']}"
            )
        lines += [
            "",
            f"Totals,,{data['total_mf']:.1f},{data['total_nf']:.1f},",
            f"NF Epidermal (micro-needling),,,,{data['total_nf_epidermal']:.1f}",
        ]

    lines += [
        "",
        f"Mode,{mode.upper()}",
        "DISCLAIMER: Values are planning estimates only.",
        "All recommendations must be reviewed by the treating clinician.",
    ]
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[DONE] CSV saved → {csv_path}")
    return csv_path


# ------------------------------------------------------------------------─────
# TERMINAL PRINT
# ------------------------------------------------------------------------─────
BOLD  = "\033[1m"; RESET = "\033[0m"; RED = "\033[91m"; DIM = "\033[2m"
ARCH_COLORS = {"T": "\033[96m", "M": "\033[92m", "J": "\033[93m", "Other": "\033[95m"}
SEV_COLORS  = {
    "None": "\033[32m", "Minimal": "\033[36m",
    "Moderate": "\033[33m", "Significant": "\033[91m", "Severe": "\033[31m",
}

def print_results(anat_data, age, image_path):
    """Print anatomic regions results table (HTML-matching format)."""
    print(f"\n{BOLD}{'─'*78}{RESET}")
    print(f"{BOLD}  ANATOMIC REGIONS  |  {os.path.basename(image_path)}  |  Age: {age}{RESET}")
    print(f"{'─'*78}")

    nf_epi  = anat_data.get("total_nf_epidermal", 0.0)
    mfa     = anat_data.get("mf_arch", {})

    print(f"  {BOLD}MF Total      : {anat_data['total_mf']:.1f} cc{RESET}")
    if mfa:
        print(f"  {BOLD}MF by Arch    : J={mfa.get('J',0):.1f}  M={mfa.get('M',0):.1f}  T={mfa.get('T',0):.1f}  Other={mfa.get('Other',0):.1f}{RESET}")
    print(f"  {BOLD}NF Deep (J+M) : {anat_data.get('nf_jm', anat_data['total_nf_deep']):.1f} cc{RESET}")
    print(f"  {BOLD}NF Epidermal  : {nf_epi:.1f} cc{RESET}  (whole-face micro-needling)")
    print(f"  Face accuracy : {int(anat_data['confidence']*100)}%   Yaw: {anat_data['yaw']:.2f}")
    print(f"{'─'*78}")
    print(f"  {'Area':<28} {'Side':<10} {'MF (cc)':>8} {'NF Deep (cc)':>13} {'Arch':>6}")
    print(f"  {'─'*76}")

    arch_order = [
        {"key": "J",     "title": "J Arch \u2014 Jawline / Lower Face"},
        {"key": "M",     "title": "M Arch \u2014 Midface"},
        {"key": "T",     "title": "T Arch \u2014 Temples / Upper Face"},
        {"key": "Other", "title": "Other \u2014 Nose / Lips / Misc"},
    ]

    rows = anat_data.get("rows", [])
    for ai in arch_order:
        arch_rows = [r for r in rows if r["arch"] == ai["key"]]
        if not arch_rows:
            continue
        ac = ARCH_COLORS[ai["key"]]
        print(f"\n{ac}{BOLD}  {ai['title']}{RESET}")

        nz = [r for r in arch_rows if r["mf"] > 0]
        zr = [r for r in arch_rows if r["mf"] == 0]
        for r in nz + zr:
            col = ac if r["mf"] > 0 else DIM
            mf_str = f"{r['mf']:.1f} cc"
            nf_str = f"{r['nf']:.1f} cc"
            print(f"  {col}{r['area']:<28}{RESET}"
                  f" {r['side']:<10}"
                  f" {mf_str:>8}"
                  f" {nf_str:>13}"
                  f" {r['arch']:>6}")

    # Totals
    print(f"\n  {'─'*76}")
    print(f"  {BOLD}{'Totals':<28}{RESET}"
          f" {'':10}"
          f" {anat_data['total_mf']:.1f} cc"
          f"    {anat_data['total_nf']:.1f} cc")
    print(f"\n  {BOLD}NF Epidermal (whole-face micro-needling): {nf_epi:.1f} cc{RESET}  \u2022 not allocated to specific regions")
    print(f"\n  Volumes are suggestions based on 2/3 correlations and selected priorities.")
    print(f"  Always adapt to anatomy, asymmetries, and clinical judgement.")
    print(f"{'─'*78}\n")


def print_jmt_results(jmt_data, age, image_path):
    """Print JMT injection points results table."""
    print(f"\n{BOLD}{'─'*76}{RESET}")
    print(f"{BOLD}  JMT INJECTION POINTS  |  {os.path.basename(image_path)}  |  Age: {age}{RESET}")
    print(f"{'─'*76}")

    nf_deep = jmt_data.get("total_nf_deep",      jmt_data["total_nf"])
    nf_epi  = jmt_data.get("total_nf_epidermal", 0.0)
    mfa     = jmt_data.get("mf_arch", {})

    print(f"  {BOLD}MF Total      : {jmt_data['total_mf']:.1f} cc{RESET}")
    if mfa:
        print(f"  {BOLD}MF by Arch    : J={mfa.get('J',0):.1f}  M={mfa.get('M',0):.1f}  T={mfa.get('T',0):.1f}  Other={mfa.get('Other',0):.1f}{RESET}")
    print(f"  {BOLD}NF Deep (J+M) : {jmt_data.get('nf_jm', nf_deep):.1f} cc{RESET}")
    print(f"  {BOLD}NF Epidermal  : {nf_epi:.1f} cc{RESET}  (whole-face micro-needling)")
    print(f"  Face accuracy : {int(jmt_data['confidence']*100)}%   Yaw: {jmt_data['yaw']:.2f}")
    print(f"{'─'*76}")
    print(f"  {'Point':<32} {'Side':<8} {'MF':>6} {'NF':>6} {'Hollow':>7} {'AI-Rec':>7}  Severity")
    print(f"  {'─'*74}")

    arch_names = {
        "T": "T Arch - Temples / Upper Face",
        "M": "M Arch - Midface",
        "J": "J Arch - Jawline / Lower Face",
        "Other": "Other - Nose / Lips / Misc",
    }
    for arch in ["J", "M", "T", "Other"]:
        ac = ARCH_COLORS[arch]
        print(f"\n{ac}{BOLD}  {arch_names[arch]}{RESET}")
        for key, v in jmt_data["regions"].items():
            if v["arch"] != arch:
                continue
            sc  = SEV_COLORS.get(v["severity"], "")
            tag = f"  {RED}! {v['severity']}{RESET}" if v["hollow"] > 0.45 else ""
            cc_str  = str(v["cc_add"]) if v["cc_add"] else "0.0"
            side = v.get("side", "Midline")
            print(f"  {ac}{v['label']:<32}{RESET}"
                  f" {side:<8}"
                  f" {v['mf']:>6.1f} {v['nf']:>6.1f}"
                  f" {v['hollow']:>7.2f}"
                  f"  {sc}{cc_str:>7}{RESET}"
                  f"  {sc}{v['severity']}{RESET}"
                  f"{tag}")

    print(f"\n  {'─'*74}")
    print(f"  {'Whole Face - NF Epidermal (micro-needling)':<46} {nf_epi:>8.1f} cc")
    print(f"\n{BOLD}  DISCLAIMER: All recommendations are planning estimates only.")
    print(f"  Clinical review by treating physician is mandatory before any procedure.{RESET}")
    print(f"{'─'*76}\n")


# ------------------------------------------------------------------------─────
# MAIN
# ------------------------------------------------------------------------─────
def select_anatomic_regions():
    """Interactive anatomic region selection."""
    print(f"\n{BOLD}  ANATOMIC REGIONS - Select Priority Areas{RESET}")
    print(f"  Unselected areas are recorded as 0 cc. Paired regions output equal L/R volumes.")
    print(f"{'─'*64}")

    for i, r in enumerate(ANATOMIC_SELECTABLE):
        pair_tag = "paired" if r["paired"] else "midline"
        print(f"  [{i+1:>2}] {r['label']:<32} ({r['arch']}, {pair_tag})")

    print(f"{'─'*64}")
    print(f"  [A] Select ALL    [N] Select NONE")
    print(f"  Enter numbers separated by commas, e.g.: 1,3,5,9")

    while True:
        raw = input("\n  Selection: ").strip().upper()
        if raw == "A":
            return {r["key"] for r in ANATOMIC_SELECTABLE}
        elif raw == "N":
            print("  [!] At least one region must be selected.")
            continue
        else:
            try:
                nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
                valid = {ANATOMIC_SELECTABLE[n-1]["key"] for n in nums
                         if 1 <= n <= len(ANATOMIC_SELECTABLE)}
                if not valid:
                    print("  [!] No valid selections. Try again.")
                    continue
                print(f"  Selected {len(valid)} regions:")
                for r in ANATOMIC_SELECTABLE:
                    if r["key"] in valid:
                        print(f"    ✓ {r['label']}")
                return valid
            except ValueError:
                print("  [!] Invalid input. Enter numbers or 'A' for all.")


def select_jmt_regions():
    """Interactive JMT region selection."""
    print(f"\n{BOLD}  JMT INJECTION POINTS - Select Priority Areas{RESET}")
    print(f"  All points are midline. Unselected = 0 cc.")
    print(f"{'─'*60}")

    for i, r in enumerate(JMT_REGIONS):
        print(f"  [{i+1:>2}] {r['label']:<36} ({r['arch']})")

    print(f"{'─'*60}")
    print(f"  [A] Select ALL    [N] Select NONE")
    print(f"  Enter numbers separated by commas, e.g.: 1,3,5,9")

    while True:
        raw = input("\n  Selection: ").strip().upper()
        if raw == "A":
            return {r["key"] for r in JMT_REGIONS}
        elif raw == "N":
            print("  [!] At least one region must be selected.")
            continue
        else:
            try:
                nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
                valid = {JMT_REGIONS[n-1]["key"] for n in nums if 1 <= n <= len(JMT_REGIONS)}
                if not valid:
                    print("  [!] No valid selections. Try again.")
                    continue
                # Show selected
                print(f"  Selected {len(valid)} points:")
                for r in JMT_REGIONS:
                    if r["key"] in valid:
                        print(f"    ✓ {r['label']}")
                return valid
            except ValueError:
                print("  [!] Invalid input. Enter numbers or 'A' for all.")


def main():
    print("=" * 60)
    print("   FAT GRAFT PREDICTOR v3 - Three-Signal Fusion")
    print("=" * 60)

    # Try to load MiDaS
    _try_load_midas()

    # ── Image path ────────────────────────────────────────────────────────
    while True:
        image_path = input("\nEnter path to face image (JPG/PNG): ").strip().strip('"').strip("'")
        if os.path.isfile(image_path):
            break
        print(f"  [!] File not found: '{image_path}'")

    # ── Age ───────────────────────────────────────────────────────────────
    while True:
        try:
            age = int(input("Enter patient age: ").strip())
            if 1 <= age <= 120:
                break
            print("  [!] Age must be 1–120.")
        except ValueError:
            print("  [!] Please enter a valid number.")

    # ── Mode selection ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Select output mode:")
    print(f"  [1] Anatomic regions  (default)")
    print(f"  [2] JMT injection points")
    print(f"  [3] Both")
    while True:
        mode_input = input("  Mode [1/2/3]: ").strip()
        if mode_input in ("", "1"):
            run_anatomic = True
            run_jmt = False
            break
        elif mode_input == "2":
            run_anatomic = False
            run_jmt = True
            break
        elif mode_input == "3":
            run_anatomic = True
            run_jmt = True
            break
        else:
            print("  [!] Enter 1, 2, or 3.")

    # ── Anatomic selection (if needed) ────────────────────────────────────
    selected_anat = None
    if run_anatomic:
        selected_anat = select_anatomic_regions()

    # ── JMT selection (if needed) ─────────────────────────────────────────
    selected_jmt = None
    if run_jmt:
        selected_jmt = select_jmt_regions()

    # ── Load image ────────────────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"[ERROR] Could not read image: {image_path}")

    MAX_DIM = 960
    h, w    = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        h, w  = frame.shape[:2]
    print(f"\n[INFO] Image: {w}x{h}  |  Age: {age}")

    # ── MediaPipe FaceMesh ────────────────────────────────────────────────
    print("[INFO] Detecting landmarks...")
    mp_face_mesh = mp_solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode        = True,
        max_num_faces            = 1,
        refine_landmarks         = True,
        min_detection_confidence = 0.5,
    ) as face_mesh:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        sys.exit("[ERROR] No face detected.\n"
                 "Tips: frontal photo, good lighting, unobstructed face.")

    landmarks = results.multi_face_landmarks[0].landmark
    print(f"[INFO] {len(landmarks)} landmarks detected.")

    # ── Compute base anatomic volumes + all three signals ─────────────────
    fat_data = calc_fat_volumes(age, landmarks, frame, h, w)

    # ── Annotated image (always uses anatomic overlay) ────────────────────
    annotated = frame.copy()
    draw_overlay(annotated, landmarks, fat_data, h, w, mp_face_mesh)

    # ═══════════════════════════════════════════════════════════════════════
    # ANATOMIC MODE
    # ═══════════════════════════════════════════════════════════════════════
    if run_anatomic:
        anat_data = calc_anatomic_volumes(age, fat_data, selected_anat)
        print_results(anat_data, age, image_path)
        panel   = build_stats_panel(anat_data, age, h)
        display_anat = np.hstack([annotated, panel])

        base, _ = os.path.splitext(image_path)
        out_img = f"{base}_result_v3.jpg"
        cv2.imwrite(out_img, display_anat, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[DONE] Anatomic image saved → {out_img}")
        export_csv(anat_data, age, image_path, mode="anatomic")

    # ═══════════════════════════════════════════════════════════════════════
    # JMT MODE
    # ═══════════════════════════════════════════════════════════════════════
    if run_jmt:
        jmt_data = calc_jmt_volumes(age, fat_data, selected_jmt)
        print_jmt_results(jmt_data, age, image_path)
        jmt_panel   = build_jmt_stats_panel(jmt_data, age, h)
        display_jmt = np.hstack([annotated, jmt_panel])

        base, _ = os.path.splitext(image_path)
        out_jmt = f"{base}_result_v3_jmt.jpg"
        cv2.imwrite(out_jmt, display_jmt, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[DONE] JMT image saved → {out_jmt}")
        export_csv(jmt_data, age, image_path, mode="jmt")

    # ── Show windows ──────────────────────────────────────────────────────
    if run_anatomic and run_jmt:
        cv2.imshow("Fat Graft Predictor v3 - Anatomic", display_anat)
        cv2.imshow("Fat Graft Predictor v3 - JMT", display_jmt)
    elif run_anatomic:
        cv2.imshow("Fat Graft Predictor v3", display_anat)
    elif run_jmt:
        cv2.imshow("Fat Graft Predictor v3 - JMT", display_jmt)

    print("[INFO] Output complete. View figures above.")
    print("[INFO] Press any key (in terminal or image window) to close.")
    
    # Unified win/terminal wait
    while True:
        # Check OpenCV window (1ms wait)
        if cv2.waitKey(1) != -1:
            break
            
        # Check terminal key press (Windows)
        if HAS_MSVCRT and msvcrt.kbhit():
            msvcrt.getch() # consume the key
            break
            
    cv2.destroyAllWindows()
    print("[INFO] Script finished.")


if __name__ == "__main__":
    main()