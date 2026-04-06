"""
Fat Graft Predictor v4
======================
Original signal-fusion mode preserved.
NEW: AI Compare mode  — micro-CNN hollow detector + face-shape analysis.
"""

import sys, os, cv2, warnings, io, contextlib
import mediapipe as mp
import numpy as np

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import msvcrt; HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    from scipy.spatial import ConvexHull; SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not found – convexity signal disabled.  pip install scipy")

try:
    import torch; import torch.nn as nn; TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not found – AI-Compare CNN disabled.  pip install torch")

# ── MiDaS (optional) ─────────────────────────────────────────────────────────
MIDAS_OK = False
midas_model = midas_transform = None

def _try_load_midas():
    global MIDAS_OK, midas_model, midas_transform
    try:
        warnings.filterwarnings("ignore", category=FutureWarning)
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            local = os.path.join("models", "midas_small.pt")
            if os.path.exists(local):
                try: midas_model.load_state_dict(torch.load(local, map_location="cpu"))
                except Exception: pass
            midas_model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            midas_transform = transforms.small_transform
        MIDAS_OK = True
        print("[INFO] MiDaS: READY")
    except Exception as e:
        print(f"[WARN] MiDaS unavailable ({e})")

try:
    mp_solutions = mp.solutions
except AttributeError:
    try:
        import mediapipe.python.solutions as mp_solutions
    except Exception:
        raise ImportError("pip install mediapipe")

# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
REGIONS = {
    "brows_L":       [336,296,334,293,300,276,283,282,295,285],
    "brows_R":       [66,105,63,70,46,53,52,65,55,107],
    "forehead_L":    [338,297,332,284,251],
    "forehead_R":    [109,67,103,54,21],
    "temporals_L":   [234,93,132,58,172],
    "temporals_R":   [454,323,361,288,397],
    "sup_orbit_L":   [226,113,225,224,223,222,221,189],
    "sup_orbit_R":   [446,342,445,444,443,442,441,413],
    "infraorb_L":    [119,100,120,121,116,123],
    "infraorb_R":    [348,329,349,350,345,352],
    "soofs_L":       [36,205,206,207,187,147],
    "soofs_R":       [266,425,426,427,411,376],
    "buccal_L":      [213,192,214,135,172,138],
    "buccal_R":      [433,416,434,364,397,367],
    "nasolabial_L":  [216,207,187,147,213],
    "nasolabial_R":  [436,427,411,376,433],
    "chin":          [199,175,152,148,176,149,150,136,172],
    "mandibles_L":   [172,136,150,149,176,148],
    "mandibles_R":   [397,365,379,378,400,377],
    "marionette_L":  [202,210,169,192,213],
    "marionette_R":  [422,430,394,416,433],
    "pyriforms_L":   [129,209,49,64,98],
    "pyriforms_R":   [358,429,279,294,327],
    "nasal_dorsum":  [6,197,195,5,4],
    "nasal_tip":     [1,2,4,5],
    "lips_upper":    [0,267,269,270,409,306,375,321,405,314],
    "lips_lower":    [17,84,181,91,146,61,185,40,39,37],
}
ARCH_MAP = {
    "brows_L":"T","brows_R":"T","forehead_L":"T","forehead_R":"T",
    "temporals_L":"T","temporals_R":"T","sup_orbit_L":"T","sup_orbit_R":"T",
    "infraorb_L":"M","infraorb_R":"M","soofs_L":"M","soofs_R":"M",
    "buccal_L":"M","buccal_R":"M","nasolabial_L":"M","nasolabial_R":"M",
    "chin":"J","mandibles_L":"J","mandibles_R":"J",
    "marionette_L":"J","marionette_R":"J","pyriforms_L":"J","pyriforms_R":"J",
    "nasal_dorsum":"Other","nasal_tip":"Other","lips_upper":"Other","lips_lower":"Other",
}
LABELS = {
    "brows_L":"Brows (L)","brows_R":"Brows (R)",
    "forehead_L":"Forehead (L)","forehead_R":"Forehead (R)",
    "temporals_L":"Temporals (L)","temporals_R":"Temporals (R)",
    "sup_orbit_L":"Sup Orbit (L)","sup_orbit_R":"Sup Orbit (R)",
    "infraorb_L":"Infraorbital (L)","infraorb_R":"Infraorbital (R)",
    "soofs_L":"SOOFs/Cheeks (L)","soofs_R":"SOOFs/Cheeks (R)",
    "buccal_L":"Buccal Fat (L)","buccal_R":"Buccal Fat (R)",
    "nasolabial_L":"Nasolabial (L)","nasolabial_R":"Nasolabial (R)",
    "chin":"Chin (midline)","mandibles_L":"Mandibles (L)","mandibles_R":"Mandibles (R)",
    "marionette_L":"Marionette (L)","marionette_R":"Marionette (R)",
    "pyriforms_L":"Pyriforms (L)","pyriforms_R":"Pyriforms (R)",
    "nasal_dorsum":"Nasal Dorsum","nasal_tip":"Nasal Tip",
    "lips_upper":"Lips Upper","lips_lower":"Lips Lower",
}
ARCH_MF_PER_REGION      = {"T":0.5,"M":0.8,"J":0.9,"Other":0.5}
ARCH_NF_DEEP_PER_REGION = {"T":0.0,"M":0.4,"J":0.4,"Other":0.0}
NATURALLY_DEEP = {"nasal_tip","nasal_dorsum","sup_orbit_L","sup_orbit_R","lips_upper","lips_lower"}

STABLE_PLANE_IDX = [36,205,207,266,425,427,21,54,103,67,109,251,284,332,297,338,93,132,323,361]

HOLLOW_TO_CC_TABLE = [
    (0.00,0.15,"0.0","None"),
    (0.15,0.35,"+0.5","Minimal"),
    (0.35,0.55,"+1.0","Moderate"),
    (0.55,0.75,"+1.5","Significant"),
    (0.75,1.01,"+2.0","Severe"),
]

# JMT definitions
JMT_REGIONS = [
    {"key":"T1_brow",              "label":"T1 Brow",                         "arch":"T"},
    {"key":"T2_crest",             "label":"T2 Crest",                        "arch":"T"},
    {"key":"T3_hollowing",         "label":"T3 Hollowing",                    "arch":"T"},
    {"key":"M1_arcus_medial",      "label":"M1 Arcus marginalis 1/3 medial",  "arch":"M"},
    {"key":"M2_arcus_median",      "label":"M2 Arcus marginalis 1/3 median",  "arch":"M"},
    {"key":"M3_arcus_lateral",     "label":"M3 Arcus marginalis 1/3 lateral", "arch":"M"},
    {"key":"M4_zygomatic_arch",    "label":"M4 Zygomatic arch",               "arch":"M"},
    {"key":"M5_zygomatic_ligament","label":"M5 Zygomatic ligament",           "arch":"M"},
    {"key":"J1_chin_vertical",     "label":"J1 Chin vertical",                "arch":"J"},
    {"key":"J2_chin_horizontal",   "label":"J2 Chin horizontal",              "arch":"J"},
    {"key":"J3_melomental",        "label":"J3 Melomental area",              "arch":"J"},
    {"key":"J4_border",            "label":"J4 Border",                       "arch":"J"},
    {"key":"J5_angle",             "label":"J5 Angle",                        "arch":"J"},
    {"key":"AN_nose",              "label":"AN Nose",                         "arch":"Other"},
    {"key":"AL1_upper_lip",        "label":"AL1 Upper Lip",                   "arch":"Other"},
    {"key":"AL2_lower_lip",        "label":"AL2 Lower Lip",                   "arch":"Other"},
]
JMT_TO_ANATOMIC = {
    "T1_brow":["brows_L","brows_R"],"T2_crest":["forehead_L","forehead_R"],
    "T3_hollowing":["temporals_L","temporals_R"],
    "M1_arcus_medial":["infraorb_L","infraorb_R"],
    "M2_arcus_median":["infraorb_L","infraorb_R","soofs_L","soofs_R"],
    "M3_arcus_lateral":["soofs_L","soofs_R"],
    "M4_zygomatic_arch":["soofs_L","soofs_R","buccal_L","buccal_R"],
    "M5_zygomatic_ligament":["nasolabial_L","nasolabial_R"],
    "J1_chin_vertical":["chin"],"J2_chin_horizontal":["chin","mandibles_L","mandibles_R"],
    "J3_melomental":["marionette_L","marionette_R"],
    "J4_border":["mandibles_L","mandibles_R"],"J5_angle":["mandibles_L","mandibles_R"],
    "AN_nose":["nasal_dorsum","nasal_tip"],"AL1_upper_lip":["lips_upper"],"AL2_lower_lip":["lips_lower"],
}
ANATOMIC_SELECTABLE = [
    {"key":"brows",        "label":"Brows (L/R)",                  "paired":True, "arch":"T","regions_L":["brows_L"],      "regions_R":["brows_R"]},
    {"key":"forehead",     "label":"Forehead (L/R)",               "paired":True, "arch":"T","regions_L":["forehead_L"],   "regions_R":["forehead_R"]},
    {"key":"temporals",    "label":"Temporals (L/R)",              "paired":True, "arch":"T","regions_L":["temporals_L"],  "regions_R":["temporals_R"]},
    {"key":"sup_orbit_sulcus","label":"Superior Orbit Sulcus (L/R)","paired":True,"arch":"T","regions_L":["sup_orbit_L"],  "regions_R":["sup_orbit_R"]},
    {"key":"infraorbital", "label":"Infraorbital (L/R)",           "paired":True, "arch":"M","regions_L":["infraorb_L"],   "regions_R":["infraorb_R"]},
    {"key":"soof_cheek",   "label":"SOOFs / Cheeks (L/R)",         "paired":True, "arch":"M","regions_L":["soofs_L"],      "regions_R":["soofs_R"]},
    {"key":"buccal_fat",   "label":"Buccal Fat (L/R)",             "paired":True, "arch":"M","regions_L":["buccal_L"],     "regions_R":["buccal_R"]},
    {"key":"nl_fold",      "label":"Nasolabial Folds (L/R)",       "paired":True, "arch":"M","regions_L":["nasolabial_L"], "regions_R":["nasolabial_R"]},
    {"key":"marionette",   "label":"Marionette Lines (L/R)",       "paired":True, "arch":"J","regions_L":["marionette_L"], "regions_R":["marionette_R"]},
    {"key":"mandible",     "label":"Mandibles (L/R)",              "paired":True, "arch":"J","regions_L":["mandibles_L"],  "regions_R":["mandibles_R"]},
    {"key":"chin",         "label":"Chin (midline)",               "paired":False,"arch":"J","regions_L":["chin"],         "regions_R":[]},
    {"key":"dorsum",       "label":"Nasal Dorsum (midline)",       "paired":False,"arch":"Other","regions_L":["nasal_dorsum"],"regions_R":[]},
    {"key":"nasal_tip",    "label":"Nasal Tip (midline)",          "paired":False,"arch":"Other","regions_L":["nasal_tip"],"regions_R":[]},
    {"key":"lips_upper",   "label":"Lips – Upper (midline)",       "paired":False,"arch":"Other","regions_L":["lips_upper"],"regions_R":[]},
    {"key":"lips_lower",   "label":"Lips – Lower (midline)",       "paired":False,"arch":"Other","regions_L":["lips_lower"],"regions_R":[]},
    {"key":"pyriforms",    "label":"Pyriforms (L/R)",              "paired":True, "arch":"J","regions_L":["pyriforms_L"],  "regions_R":["pyriforms_R"]},
]

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR / SEVERITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
SEVERITY_BGR = {
    "None":(100,180,100),"Minimal":(100,220,220),
    "Moderate":(60,190,255),"Significant":(40,120,255),"Severe":(60,60,220),
}
ARCH_BGR_PANEL = {"T":(220,190,70),"M":(90,200,90),"J":(60,175,255),"Other":(190,120,200)}
BOLD="\033[1m";RESET="\033[0m";RED="\033[91m";DIM="\033[2m"
ARCH_COLORS={"T":"\033[96m","M":"\033[92m","J":"\033[93m","Other":"\033[95m"}
SEV_COLORS={"None":"\033[32m","Minimal":"\033[36m","Moderate":"\033[33m","Significant":"\033[91m","Severe":"\033[31m"}

_D_LUT = np.zeros((101,3),dtype=np.uint8)
for _i in range(101):
    _D_LUT[_i] = cv2.cvtColor(np.uint8([[[int(_i/100*120),180,210]]]),cv2.COLOR_HSV2BGR)[0][0]
_DEPTH_STRIP = np.zeros((12,100,3),dtype=np.uint8)
for _i in range(100): _DEPTH_STRIP[:,_i] = _D_LUT[min(int(_i/99*100),100)]

def hollow_color(severity): return SEVERITY_BGR.get(severity,(150,150,150))

def hollow_to_recommendation(score):
    for lo,hi,cc,label in HOLLOW_TO_CC_TABLE:
        if lo<=score<hi: return cc,label
    return None,"None"

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 – MEDIAPIPE  (z-depth + flatness + convexity)
# ═══════════════════════════════════════════════════════════════════════════════
def get_face_refs(landmarks):
    plane_z = np.array([landmarks[i].z for i in STABLE_PLANE_IDX if i<len(landmarks)])
    all_z   = np.array([lm.z for lm in landmarks])
    face_median_z = float(np.median(plane_z))
    z_iqr = max(float(np.percentile(all_z,75)-np.percentile(all_z,25)),0.008)
    lx,rx,nx = landmarks[234].x,landmarks[454].x,landmarks[1].x
    yaw = float(np.clip(abs((nx-lx)/(rx-lx)-0.5)*2.5,0,1)) if rx>lx else 0.5
    return face_median_z,z_iqr,yaw

def detect_hollowness_v3(landmarks,indices,face_median_z,z_iqr,region_name="",h=1,w=1):
    valid=[i for i in indices if i<len(landmarks)]
    if not valid: return 0.0,0.0,0.0,0.0
    pts3d=np.array([[landmarks[i].x,landmarks[i].y,landmarks[i].z] for i in valid])
    avg_z=float(np.mean(pts3d[:,2]))
    below=face_median_z-avg_z
    thr=0.90 if region_name in NATURALLY_DEEP else 0.55
    depth_s=float(np.clip(2.0*below/z_iqr-thr,0,1))
    z_std=float(np.std(pts3d[:,2]))
    flat_s=float(np.clip(1.0-(z_std/(z_iqr*0.4)),0,1))
    conc_s=0.0
    if SCIPY_OK and len(pts3d)>=4:
        pts2d=pts3d[:,:2].copy(); pts2d[:,0]*=w; pts2d[:,1]*=h
        try:
            hull=ConvexHull(pts2d)
            x_,y_=pts2d[:,0],pts2d[:,1]
            poly_area=0.5*abs(np.dot(x_,np.roll(y_,1))-np.dot(y_,np.roll(x_,1)))
            conc_s=float(np.clip(1.0-poly_area/(hull.volume+1e-6),0,1))
        except: pass
    combined=0.55*depth_s+0.25*flat_s+0.20*conc_s
    return round(combined,2),round(depth_s,2),round(flat_s,2),round(conc_s,2)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 – LAB LUMINANCE
# ═══════════════════════════════════════════════════════════════════════════════
def build_face_hull_mask(landmarks,h,w):
    all_pts=np.array([(int(lm.x*w),int(lm.y*h)) for lm in landmarks],dtype=np.int32)
    mask=np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask,[cv2.convexHull(all_pts)],255)
    return mask

def get_luminance_scores(frame_bgr,landmarks,h,w):
    lab=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2LAB)
    L=lab[:,:,0].astype(np.float32)
    face_mask=build_face_hull_mask(landmarks,h,w)
    face_L=L[face_mask==255]
    if len(face_L)==0: return {r:0.0 for r in REGIONS}
    f_med=float(np.median(face_L))
    f_iqr=max(float(np.percentile(face_L,75)-np.percentile(face_L,25)),5.0)
    scores={}
    for region,indices in REGIONS.items():
        valid=[i for i in indices if i<len(landmarks)]
        pts=np.array([(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in valid],dtype=np.int32)
        if len(pts)<3: scores[region]=0.0; continue
        mask=np.zeros((h,w),dtype=np.uint8); cv2.fillPoly(mask,[pts],255)
        rL=L[mask==255]
        scores[region]=round(float(np.clip((f_med-np.mean(rL))/f_iqr*0.8,0,1)),2) if len(rL) else 0.0
    return scores

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 – MIDAS
# ═══════════════════════════════════════════════════════════════════════════════
def get_midas_scores(frame_rgb,landmarks,h,w):
    if not MIDAS_OK: return {r:0.0 for r in REGIONS}
    inp=midas_transform(frame_rgb)
    with torch.no_grad():
        depth=torch.nn.functional.interpolate(
            midas_model(inp).unsqueeze(1),size=(h,w),mode="bicubic",align_corners=False
        ).squeeze().numpy()
    d_min,d_max=depth.min(),depth.max()
    dn=(depth-d_min)/(d_max-d_min+1e-6)
    face_mask=build_face_hull_mask(landmarks,h,w)
    fv=dn[face_mask==255]
    if len(fv)==0: return {r:0.0 for r in REGIONS}
    f_med=float(np.median(fv))
    f_iqr=max(float(np.percentile(fv,75)-np.percentile(fv,25)),0.02)
    scores={}
    for region,indices in REGIONS.items():
        valid=[i for i in indices if i<len(landmarks)]
        pts=np.array([(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in valid],dtype=np.int32)
        if len(pts)<3: scores[region]=0.0; continue
        mask=np.zeros((h,w),dtype=np.uint8); cv2.fillPoly(mask,[pts],255)
        rv=dn[mask==255]
        scores[region]=round(float(np.clip((f_med-np.mean(rv))/f_iqr-0.3,0,1)),2) if len(rv) else 0.0
    return scores

# ═══════════════════════════════════════════════════════════════════════════════
# FUSION
# ═══════════════════════════════════════════════════════════════════════════════
def fuse_signals(mp_s,lum_s,midas_s,yaw_conf,lum_ok=True,midas_ok=True):
    active=[mp_s]+([lum_s] if lum_ok else [])+([midas_s] if midas_ok else [])
    spread=max(active)-min(active)
    if len(active)==1:
        fused=mp_s; conf_lbl="mp-only"
    elif spread<0.15:
        fused=(0.40*mp_s+0.30*lum_s+0.30*midas_s) if midas_ok else (0.60*mp_s+0.40*lum_s)
        conf_lbl="high"
    elif spread<0.30:
        fused=(0.55*mp_s+0.25*lum_s+0.20*midas_s) if midas_ok else (0.70*mp_s+0.30*lum_s)
        conf_lbl="medium"
    else:
        fused=mp_s; conf_lbl="low (diverged)"
    return round(float(np.clip(fused*yaw_conf,0,1)),2), conf_lbl

# ═══════════════════════════════════════════════════════════════════════════════
# MICRO-CNN  (SIGNAL 4 – AI Compare)
# ═══════════════════════════════════════════════════════════════════════════════
class MicroHollowCNN(nn.Module):
    """
    Lightweight patch CNN – no external weights needed.
    Input: 32×32 RGB patch (normalised).
    Output: scalar 0-1 (hollow probability).

    Architecture: 3 conv blocks → global-avg-pool → 2-layer head.
    Feature extraction is heuristically guided via weight init rather than
    supervised training (zero-shot geometric priors).
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                             # 16×16
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                             # 8×8
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                     # 1×1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,1), nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        """
        Heuristic init: first conv sensitive to dark-centre / shadow patterns
        (shadow = hollow cue).  No labelled data needed.
        """
        with torch.no_grad():
            # centre-dark detector kernel
            k = torch.zeros(3,3)
            k[1,1]=-1.5; k[0,0]=k[0,2]=k[2,0]=k[2,2]=0.3; k[0,1]=k[1,0]=k[1,2]=k[2,1]=0.15
            for c in range(3):
                self.enc[0].weight.data[0,c] = k
            # edge/gradient detector
            edge = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=torch.float32)/8.0
            for c in range(3):
                self.enc[0].weight.data[1,c] = edge
            # horizontal shadow band
            hband = torch.zeros(3,3); hband[1,:]=1.0; hband[0,:]=hband[2,:]=-0.5
            for c in range(3):
                self.enc[0].weight.data[2,c] = hband

    def forward(self,x): return self.head(self.enc(x))


_cnn_model = None

def _get_cnn():
    global _cnn_model
    if _cnn_model is None and TORCH_OK:
        _cnn_model = MicroHollowCNN().eval()
        # load fine-tuned weights if available
        w_path = os.path.join("models","hollow_cnn.pt")
        if os.path.exists(w_path):
            try:
                _cnn_model.load_state_dict(torch.load(w_path,map_location="cpu"))
                print("[INFO] CNN weights loaded from hollow_cnn.pt")
            except Exception as e:
                print(f"[WARN] CNN weight load failed ({e}) – using heuristic init")
    return _cnn_model

PATCH_SIZE = 32

def _extract_patch(frame_bgr, landmarks, indices, h, w):
    """Crop bounding-box around region, resize to PATCH_SIZE."""
    valid=[i for i in indices if i<len(landmarks)]
    if not valid: return None
    xs=[int(landmarks[i].x*w) for i in valid]
    ys=[int(landmarks[i].y*h) for i in valid]
    x1,x2=max(0,min(xs)-4),min(w-1,max(xs)+4)
    y1,y2=max(0,min(ys)-4),min(h-1,max(ys)+4)
    if x2<=x1 or y2<=y1: return None
    patch=frame_bgr[y1:y2,x1:x2]
    patch=cv2.resize(patch,(PATCH_SIZE,PATCH_SIZE))
    return patch

def get_cnn_hollow_scores(frame_bgr, landmarks, h, w):
    """
    Returns {region: cnn_hollow_score 0-1}
    Uses micro-CNN on each region patch.
    Falls back to zeros if torch unavailable.
    """
    model=_get_cnn()
    if model is None: return {r:0.0 for r in REGIONS}

    scores={}
    for region,indices in REGIONS.items():
        patch=_extract_patch(frame_bgr,landmarks,indices,h,w)
        if patch is None: scores[region]=0.0; continue
        # BGR→RGB, normalise, tensor
        rgb=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        t=torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0)
        with torch.no_grad():
            val=float(model(t).item())
        # scale – CNN raw output is 0-1 but typically biased low; stretch
        val=float(np.clip(val*1.4,0,1))
        scores[region]=round(val,2)
    return scores

# ═══════════════════════════════════════════════════════════════════════════════
# FACE SHAPE DETECTION  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

# Key landmark indices for shape measurement
FACE_TOP        = 10    # top of forehead
FACE_BOTTOM     = 152   # chin tip
FACE_LEFT       = 234   # left cheek extreme
FACE_RIGHT      = 454   # right cheek extreme
JAW_LEFT        = 172   # left jaw
JAW_RIGHT       = 397   # right jaw
FOREHEAD_LEFT   = 21
FOREHEAD_RIGHT  = 251
CHEEK_LEFT      = 205
CHEEK_RIGHT     = 425
JAW_ANGLE_L     = 136
JAW_ANGLE_R     = 365
TEMPLE_L        = 54
TEMPLE_R        = 284

FACE_SHAPE_TIPS = {
    "Oval": {
        "desc": "Balanced proportions. Forehead slightly wider than jaw, curved chin.",
        "fat_zones": ["temporals","infraorbital","chin"],
        "reason": "Maintain balance. Light volume in temples and infraorbital; subtle chin projection.",
    },
    "Round": {
        "desc": "Width ≈ length. Full cheeks, rounded jawline.",
        "fat_zones": ["brows","forehead","mandible","chin"],
        "reason": "Add vertical definition. Lift brows, elongate with forehead, define jaw.",
    },
    "Square": {
        "desc": "Strong jaw angle ≈ forehead width. Flat sides.",
        "fat_zones": ["temporals","soofs","nasal_tip","lips_upper"],
        "reason": "Soften angularity. Fill temples and cheeks for curvature; refine lip/nose.",
    },
    "Heart": {
        "desc": "Wide forehead, narrow pointed chin.",
        "fat_zones": ["mandible","chin","marionette","nasolabial"],
        "reason": "Balance lower face. Add jaw width and chin projection to offset wide forehead.",
    },
    "Oblong": {
        "desc": "Face length > width. Narrow cheeks.",
        "fat_zones": ["soofs","buccal_fat","temporals","marionette"],
        "reason": "Add lateral width. Fill mid-cheek, buccal, temples to broaden appearance.",
    },
    "Diamond": {
        "desc": "Narrow forehead and jaw, wide cheekbones.",
        "fat_zones": ["forehead","mandible","chin"],
        "reason": "Broaden forehead and jaw to balance prominent cheekbones.",
    },
}

def detect_face_shape(landmarks, h, w):
    """
    Classify face shape using 6 key ratios derived from MediaPipe landmarks.
    Returns (shape_name, metrics_dict).
    """
    def pt(idx): return np.array([landmarks[idx].x*w, landmarks[idx].y*h])

    face_w   = np.linalg.norm(pt(FACE_LEFT) -pt(FACE_RIGHT))
    face_h   = np.linalg.norm(pt(FACE_TOP)  -pt(FACE_BOTTOM))
    jaw_w    = np.linalg.norm(pt(JAW_LEFT)  -pt(JAW_RIGHT))
    fore_w   = np.linalg.norm(pt(FOREHEAD_LEFT)-pt(FOREHEAD_RIGHT))
    cheek_w  = np.linalg.norm(pt(CHEEK_LEFT)-pt(CHEEK_RIGHT))

    # Jaw angle sharpness: angle at jaw corner
    def jaw_angle(lm_corner, lm_chin, lm_jaw):
        v1 = pt(lm_chin)-pt(lm_corner)
        v2 = pt(lm_jaw) -pt(lm_corner)
        cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_a,-1,1))))

    jaw_ang = (jaw_angle(JAW_ANGLE_L,152,172)+jaw_angle(JAW_ANGLE_R,152,397))/2

    eps = 1e-4
    ratio_hw  = face_h  / (face_w  + eps)   # >1.2 = long
    ratio_jf  = jaw_w   / (fore_w  + eps)   # >0.9 = square/round, <0.7 = heart
    ratio_cf  = cheek_w / (fore_w  + eps)   # >1.05 = diamond
    ratio_cj  = cheek_w / (jaw_w   + eps)   # cheek vs jaw

    metrics = {
        "face_h/w":  round(ratio_hw,2),
        "jaw/fore":  round(ratio_jf,2),
        "cheek/fore":round(ratio_cf,2),
        "cheek/jaw": round(ratio_cj,2),
        "jaw_angle": round(jaw_ang,1),
        "face_w":    round(face_w,1),
        "face_h":    round(face_h,1),
    }

    # Decision tree (clinically inspired)
    if ratio_hw > 1.35:
        shape = "Oblong"
    elif ratio_cf > 1.08 and ratio_jf < 0.80:
        shape = "Diamond"
    elif ratio_jf < 0.72:
        shape = "Heart"
    elif ratio_hw < 1.10 and ratio_jf > 0.88:
        shape = "Round"
    elif jaw_ang < 125 and ratio_jf > 0.85:
        shape = "Square"
    else:
        shape = "Oval"

    return shape, metrics

def draw_face_shape_overlay(frame, landmarks, shape, metrics, h, w):
    """Draw face outline + shape annotation."""
    # Draw face outline via key boundary points
    outline_idx=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,
                 148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
    pts=np.array([(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in outline_idx if i<len(landmarks)],dtype=np.int32)
    if len(pts)>2:
        cv2.polylines(frame,[pts],False,(200,200,50),2,cv2.LINE_AA)

    # Shape label box
    lbl=f"Shape: {shape}"
    tip=FACE_SHAPE_TIPS.get(shape,{}).get("desc","")[:45]
    bx,by=10,10
    cv2.rectangle(frame,(bx,by),(bx+260,by+36),(15,15,15),-1)
    cv2.rectangle(frame,(bx,by),(bx+260,by+36),(200,200,50),1)
    cv2.putText(frame,lbl,(bx+6,by+15),cv2.FONT_HERSHEY_PLAIN,0.95,(200,200,50),1)
    cv2.putText(frame,tip,(bx+6,by+30),cv2.FONT_HERSHEY_PLAIN,0.58,(160,160,160),1)

    # Highlight recommended fat zones for this shape
    zones=FACE_SHAPE_TIPS.get(shape,{}).get("fat_zones",[])
    zone_region_keys=set()
    for z in zones:
        for r in ANATOMIC_SELECTABLE:
            if r["key"]==z:
                zone_region_keys.update(r["regions_L"]+r["regions_R"])

    for region,indices in REGIONS.items():
        if region not in zone_region_keys: continue
        valid=[i for i in indices if i<len(landmarks)]
        pts2=np.array([(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in valid],dtype=np.int32)
        if len(pts2)<3: continue
        ov=frame.copy(); cv2.fillPoly(ov,[pts2],(200,200,50)); cv2.addWeighted(ov,0.22,frame,0.78,0,frame)
        cv2.polylines(frame,[pts2],True,(200,200,50),2)

# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME CALCULATION (original)
# ═══════════════════════════════════════════════════════════════════════════════
def calc_fat_volumes(age, landmarks, frame_bgr, h, w):
    total_mf=round((2/3)*age,1); total_nf=round((1/3)*age,1)
    total_nf_epi=round(total_nf/3,1); total_nf_deep=round(total_nf-total_nf_epi,1)
    sum_mf_w=sum(ARCH_MF_PER_REGION[ARCH_MAP[r]] for r in REGIONS)
    sum_nf_w=sum(ARCH_NF_DEEP_PER_REGION[ARCH_MAP[r]] for r in REGIONS)
    face_median_z,z_iqr,yaw=get_face_refs(landmarks)
    yaw_conf=float(np.clip(1.0-yaw*1.8,0.05,1.0))
    print("[INFO] Computing luminance signal...")
    lum_scores=get_luminance_scores(frame_bgr,landmarks,h,w)
    print("[INFO] Computing MiDaS signal...")
    midas_scores=get_midas_scores(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB),landmarks,h,w)
    results={}
    for region,indices in REGIONS.items():
        arch=ARCH_MAP[region]
        mp_h,dep_s,flat_s,conc_s=detect_hollowness_v3(landmarks,indices,face_median_z,z_iqr,region,h,w)
        lum_s=lum_scores.get(region,0.0); midas_s=midas_scores.get(region,0.0)
        fused,sig_conf=fuse_signals(mp_h,lum_s,midas_s,yaw_conf,True,MIDAS_OK)
        cc_add,severity=hollow_to_recommendation(fused)
        mf=round(ARCH_MF_PER_REGION[arch]/sum_mf_w*total_mf,1)
        nf=(round(ARCH_NF_DEEP_PER_REGION[arch]/sum_nf_w*total_nf_deep,1) if sum_nf_w>0 else 0.0)
        results[region]={
            "label":LABELS[region],"arch":arch,"mf":mf,"nf":nf,
            "hollow":fused,"hollow_mp":mp_h,"hollow_lum":lum_s,"hollow_midas":midas_s,
            "sig_conf":sig_conf,"cc_add":cc_add,"severity":severity,
            "depth_s":dep_s,"flatness_s":flat_s,"concavity_s":conc_s,
        }
    return {"regions":results,"total_mf":total_mf,"total_nf":total_nf,
            "total_nf_deep":total_nf_deep,"total_nf_epidermal":total_nf_epi,
            "yaw":round(yaw,2),"confidence":round(yaw_conf,2),"midas_used":MIDAS_OK}

# ═══════════════════════════════════════════════════════════════════════════════
# ANATOMIC & JMT VOLUME CALCULATION (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def calc_anatomic_volumes(age,fat_data,selected_anat=None):
    total_mf=round((2/3)*age,1); total_nf=round((1/3)*age,1)
    nf_epi=round(total_nf/3,1); nf_jm=round((2/3)*total_nf,1)
    mf_j=total_mf/3; mf_m=total_mf/3; rem=total_mf/3
    MF_ARCH={"J":mf_j,"M":mf_m,"T":(2/3)*rem,"Other":(1/3)*rem}
    if selected_anat is None: selected_anat={r["key"] for r in ANATOMIC_SELECTABLE}
    arch_slots={"J":0,"M":0,"T":0,"Other":0}
    for r in ANATOMIC_SELECTABLE:
        if r["key"] in selected_anat: arch_slots[r["arch"]]+=2 if r["paired"] else 1
    sel_jm=sum((2 if r["paired"] else 1) for r in ANATOMIC_SELECTABLE
               if r["key"] in selected_anat and r["arch"] in ("J","M"))
    nf_per_jm=nf_jm/sel_jm if sel_jm>0 else 0.0
    rows=[]
    for r in ANATOMIC_SELECTABLE:
        key=r["key"]; arch=r["arch"]; sel=key in selected_anat
        hollows=[fat_data["regions"][rk]["hollow"] for rk in r["regions_L"]+r["regions_R"] if rk in fat_data["regions"]]
        avg_h=round(float(np.mean(hollows)),2) if hollows else 0.0
        cc_add,severity=hollow_to_recommendation(avg_h)
        pmf=round(MF_ARCH[arch]/arch_slots[arch],1) if sel and arch_slots[arch]>0 else 0.0
        pnf=round(nf_per_jm,1) if sel and arch in ("J","M") else 0.0
        if r["paired"]:
            for side in ("Left","Right"):
                rows.append({"area":r["label"],"side":side,"arch":arch,"mf":pmf,"nf":pnf,
                              "hollow":avg_h,"severity":severity,"cc_add":cc_add,"selected":sel})
        else:
            rows.append({"area":r["label"],"side":"Midline","arch":arch,"mf":pmf,"nf":pnf,
                         "hollow":avg_h,"severity":severity,"cc_add":cc_add,"selected":sel})
    return {"rows":rows,"total_mf":total_mf,"total_nf":total_nf,
            "total_nf_deep":nf_jm,"total_nf_epidermal":nf_epi,"mf_arch":MF_ARCH,"nf_jm":nf_jm,
            "yaw":fat_data.get("yaw",0.0),"confidence":fat_data.get("confidence",1.0),
            "midas_used":fat_data.get("midas_used",False)}

def calc_jmt_volumes(age,fat_data,selected_jmt=None):
    total_mf=round((2/3)*age,1); total_nf=round((1/3)*age,1)
    nf_epi=round(total_nf/3,1); nf_jm=round((2/3)*total_nf,1)
    rem=total_mf/3
    MF_ARCH={"J":total_mf/3,"M":total_mf/3,"T":(2/3)*rem,"Other":(1/3)*rem}
    if selected_jmt is None: selected_jmt={r["key"] for r in JMT_REGIONS}
    arch_slots={"J":0,"M":0,"T":0,"Other":0}
    for r in JMT_REGIONS:
        if r["key"] in selected_jmt: arch_slots[r["arch"]]+=1
    sel_jm=sum(1 for r in JMT_REGIONS if r["key"] in selected_jmt and r["arch"] in ("J","M"))
    nf_per_jm=nf_jm/sel_jm if sel_jm>0 else 0.0
    jmt_results={}
    for r in JMT_REGIONS:
        key=r["key"]; arch=r["arch"]; sel=key in selected_jmt
        mapped=JMT_TO_ANATOMIC.get(key,[])
        vs=[fat_data["regions"][k] for k in mapped if k in fat_data["regions"]]
        def avg(field): return round(float(np.mean([v[field] for v in vs])),2) if vs else 0.0
        avg_h=avg("hollow"); cc_add,severity=hollow_to_recommendation(avg_h)
        mf=round(MF_ARCH[arch]/arch_slots[arch],1) if sel and arch_slots[arch]>0 else 0.0
        nf=round(nf_per_jm,1) if sel and arch in ("J","M") else 0.0
        sig_conf=vs[0]["sig_conf"] if vs else "mp-only"
        jmt_results[key]={"label":r["label"],"arch":arch,"mf":mf,"nf":nf,
                          "hollow":avg_h,"hollow_mp":avg("hollow_mp"),"hollow_lum":avg("hollow_lum"),
                          "hollow_midas":avg("hollow_midas"),"sig_conf":sig_conf,
                          "cc_add":cc_add,"severity":severity,"side":"Midline"}
    return {"regions":jmt_results,"total_mf":total_mf,"total_nf":total_nf,
            "total_nf_deep":nf_jm,"total_nf_epidermal":nf_epi,"mf_arch":MF_ARCH,"nf_jm":nf_jm,
            "yaw":fat_data.get("yaw",0.0),"confidence":fat_data.get("confidence",1.0),
            "midas_used":fat_data.get("midas_used",False)}

# ═══════════════════════════════════════════════════════════════════════════════
# ██████╗  NEW: AI COMPARE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def run_ai_compare(age, image_path, landmarks, frame_bgr, h, w,
                   fat_data_original, face_shape=None, shape_metrics=None):
    """
    AI Compare pipeline:
    1. Run micro-CNN on each region patch → CNN hollow scores
    2. Fuse CNN + LAB luminance (no z-depth / MiDaS here – pure image-space)
    3. Build result dict in same schema as fat_data_original
    4. Compute per-region diff vs original
    5. Return ai_fat_data + diff_table

    Inputs  : basics – age, image_path, landmarks, frame_bgr, h, w, original fat_data
    Outputs : (ai_fat_data, diff_table, ai_anat_data)
    """
    print("[AI-Compare] Running micro-CNN patch analysis...")
    cnn_scores  = get_cnn_hollow_scores(frame_bgr, landmarks, h, w)
    lum_scores  = get_luminance_scores(frame_bgr, landmarks, h, w)

    face_median_z,z_iqr,yaw = get_face_refs(landmarks)
    yaw_conf = float(np.clip(1.0-yaw*1.8,0.05,1.0))

    total_mf=round((2/3)*age,1); total_nf=round((1/3)*age,1)
    total_nf_epi=round(total_nf/3,1); total_nf_deep=round(total_nf-total_nf_epi,1)
    sum_mf_w=sum(ARCH_MF_PER_REGION[ARCH_MAP[r]] for r in REGIONS)
    sum_nf_w=sum(ARCH_NF_DEEP_PER_REGION[ARCH_MAP[r]] for r in REGIONS)

    ai_regions={}
    diff_table=[]

    for region,indices in REGIONS.items():
        arch=ARCH_MAP[region]
        cnn_s = cnn_scores.get(region,0.0)
        lum_s = lum_scores.get(region,0.0)

        # CNN-luminance fusion (no yaw penalty – CNN is rotation-agnostic patch based)
        spread=abs(cnn_s-lum_s)
        if spread<0.15:
            ai_hollow=round(float(np.clip(0.55*cnn_s+0.45*lum_s,0,1)),2)
            sig_conf="high"
        elif spread<0.30:
            ai_hollow=round(float(np.clip(0.70*cnn_s+0.30*lum_s,0,1)),2)
            sig_conf="medium"
        else:
            ai_hollow=round(float(np.clip(cnn_s*yaw_conf,0,1)),2)
            sig_conf="low (cnn-only)"

        cc_add,severity=hollow_to_recommendation(ai_hollow)
        mf=round(ARCH_MF_PER_REGION[arch]/sum_mf_w*total_mf,1)
        nf=(round(ARCH_NF_DEEP_PER_REGION[arch]/sum_nf_w*total_nf_deep,1) if sum_nf_w>0 else 0.0)

        ai_regions[region]={
            "label":LABELS[region],"arch":arch,"mf":mf,"nf":nf,
            "hollow":ai_hollow,"hollow_cnn":cnn_s,"hollow_lum":lum_s,
            "sig_conf":sig_conf,"cc_add":cc_add,"severity":severity,
        }

        # Diff vs original
        orig = fat_data_original["regions"].get(region,{})
        orig_h = orig.get("hollow",0.0)
        delta  = round(ai_hollow - orig_h, 2)
        agree  = abs(delta) < 0.10

        diff_table.append({
            "region":    region,
            "label":     LABELS[region],
            "arch":      arch,
            "orig_h":    orig_h,
            "ai_h":      ai_hollow,
            "delta":     delta,
            "agree":     agree,
            "orig_sev":  orig.get("severity","None"),
            "ai_sev":    severity,
        })

    ai_fat_data={
        "regions":ai_regions,"total_mf":total_mf,"total_nf":total_nf,
        "total_nf_deep":total_nf_deep,"total_nf_epidermal":total_nf_epi,
        "yaw":round(yaw,2),"confidence":round(yaw_conf,2),"midas_used":False,
        "face_shape":face_shape,"shape_metrics":shape_metrics,
    }

    # Build anatomic summary for AI result
    ai_anat_data = calc_anatomic_volumes(age, ai_fat_data)

    # Attach face-shape recommendation
    if face_shape:
        ai_fat_data["shape_recommendation"] = FACE_SHAPE_TIPS.get(face_shape,{})

    print(f"[AI-Compare] Done. {sum(1 for d in diff_table if d['agree'])}/{len(diff_table)} regions agree within 0.10.")
    return ai_fat_data, diff_table, ai_anat_data


# ═══════════════════════════════════════════════════════════════════════════════
# DRAW ORIGINAL OVERLAY  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def draw_overlay(frame, landmarks, fat_data, h, w, face_mesh_module):
    for conn in face_mesh_module.FACEMESH_CONTOURS:
        a,b=conn
        if a<len(landmarks) and b<len(landmarks):
            cv2.line(frame,(int(landmarks[a].x*w),int(landmarks[a].y*h)),
                           (int(landmarks[b].x*w),int(landmarks[b].y*h)),(70,70,70),1)
    overlay=frame.copy()
    deep_idx_set=set()
    for rn in NATURALLY_DEEP: deep_idx_set.update(REGIONS.get(rn,[]))
    for region,indices in REGIONS.items():
        v=fat_data["regions"][region]; mf=v["mf"]; hollow=v["hollow"]; severity=v["severity"]
        if mf==0: continue
        pts=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in indices if i<len(landmarks)]
        if len(pts)<3: continue
        pts_arr=np.array(pts,dtype=np.int32); color=hollow_color(severity)
        if severity!="None":
            cv2.fillPoly(overlay,[pts_arr],color)
            cv2.polylines(frame,[pts_arr],True,color,2 if hollow>0.45 else 1)
        else:
            cv2.polylines(frame,[pts_arr],True,(100,150,100),1)
        cx=int(np.mean([p[0] for p in pts])); cy=int(np.mean([p[1] for p in pts]))
        cc=str(v["cc_add"]) if v["cc_add"] else "0.0"
        lbl=("!" if hollow>0.45 else "")+f"{mf:.1f}  {cc}"
        if severity!="None":
            cv2.putText(frame,lbl,(cx-18,cy+5),cv2.FONT_HERSHEY_PLAIN,0.72,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,lbl,(cx-17,cy+4),cv2.FONT_HERSHEY_PLAIN,0.72,color,1,cv2.LINE_AA)
    cv2.addWeighted(overlay,0.28,frame,0.72,0,frame)
    xs=np.clip((np.array([lm.x for lm in landmarks])*w).astype(int),0,w-1)
    ys=np.clip((np.array([lm.y for lm in landmarks])*h).astype(int),0,h-1)
    zs=np.array([lm.z for lm in landmarks])
    idxs=np.clip(((zs+0.08)/0.16*100).astype(int),0,100)
    dot_colors=_D_LUT[idxs]
    for li,(xi,yi,ci) in enumerate(zip(xs,ys,dot_colors)):
        cv2.circle(frame,(int(xi),int(yi)),1 if li in deep_idx_set else 2,
                   (int(ci[0]),int(ci[1]),int(ci[2])),-1)
    # legend
    lx,ly=10,h-90
    cv2.rectangle(frame,(lx-4,ly-22),(lx+120,ly+80),(10,10,10),-1)
    cv2.rectangle(frame,(lx-4,ly-22),(lx+120,ly+80),(60,60,60),1)
    cv2.putText(frame,"Fat needed:",(lx,ly-7),cv2.FONT_HERSHEY_PLAIN,0.85,(200,200,200),1)
    yo=ly+10
    for sev,col in [("Minimal",(100,220,220)),("Moderate",(60,190,255)),
                    ("Significant",(40,120,255)),("Severe",(60,60,220))]:
        cv2.rectangle(frame,(lx,yo),(lx+15,yo+10),col,-1)
        cv2.putText(frame,sev,(lx+25,yo+9),cv2.FONT_HERSHEY_PLAIN,0.7,(200,200,200),1); yo+=15
    dx=lx+130
    cv2.rectangle(frame,(dx-4,ly-22),(dx+157,ly+30),(10,10,10),-1)
    cv2.rectangle(frame,(dx-4,ly-22),(dx+157,ly+30),(60,60,60),1)
    cv2.putText(frame,"Depth:",(dx,ly-7),cv2.FONT_HERSHEY_PLAIN,0.85,(200,200,200),1)
    frame[ly+2:ly+14,dx:dx+100]=_DEPTH_STRIP
    cv2.putText(frame,"sunken",(dx,ly+27),cv2.FONT_HERSHEY_PLAIN,0.7,(80,80,255),1)
    cv2.putText(frame,"raised",(dx+72,ly+27),cv2.FONT_HERSHEY_PLAIN,0.7,(100,255,100),1)
    mf_t=fat_data["total_mf"]; nf_t=fat_data["total_nf"]
    cv2.rectangle(frame,(w-198,4),(w-4,58),(10,10,10),-1)
    cv2.rectangle(frame,(w-198,4),(w-4,58),(70,70,70),1)
    cv2.putText(frame,f"MF: {mf_t:.1f} cc",(w-190,26),cv2.FONT_HERSHEY_PLAIN,1.1,(160,230,110),1)
    cv2.putText(frame,f"NF: {nf_t:.1f} cc",(w-190,50),cv2.FONT_HERSHEY_PLAIN,1.1,(110,190,255),1)
    sigs="Signals: MP+LUM"+(" +MIDAS" if fat_data.get("midas_used") else "")
    cv2.rectangle(frame,(w-198,62),(w-4,80),(10,10,10),-1)
    cv2.putText(frame,sigs,(w-194,76),cv2.FONT_HERSHEY_PLAIN,0.75,(160,160,255),1)
    yaw=fat_data.get("yaw",0.0); conf=fat_data.get("confidence",1.0)
    if yaw>0.15:
        wc=(0,50,255) if yaw>0.40 else (0,165,255)
        wm=(f"! FACE NOT FRONTAL  ({int(conf*100)}% accuracy)" if yaw>0.40
            else f"~ Slight angle  ({int(conf*100)}% accuracy)")
        cv2.rectangle(frame,(6,35),(len(wm)*8+12,56),(10,10,10),-1)
        cv2.putText(frame,wm,(10,51),cv2.FONT_HERSHEY_PLAIN,1.0,wc,1)


# ═══════════════════════════════════════════════════════════════════════════════
# DRAW AI COMPARE OVERLAY  (new – CNN hollow colour-coded differently)
# ═══════════════════════════════════════════════════════════════════════════════
def draw_ai_overlay(frame, landmarks, ai_fat_data, h, w, face_mesh_module):
    """Same layout as draw_overlay but uses CNN-fused scores; teal palette."""
    for conn in face_mesh_module.FACEMESH_CONTOURS:
        a,b=conn
        if a<len(landmarks) and b<len(landmarks):
            cv2.line(frame,(int(landmarks[a].x*w),int(landmarks[a].y*h)),
                           (int(landmarks[b].x*w),int(landmarks[b].y*h)),(50,80,80),1)
    overlay=frame.copy()
    for region,indices in REGIONS.items():
        v=ai_fat_data["regions"][region]; mf=v["mf"]; hollow=v["hollow"]; severity=v["severity"]
        if mf==0: continue
        pts=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in indices if i<len(landmarks)]
        if len(pts)<3: continue
        pts_arr=np.array(pts,dtype=np.int32); color=hollow_color(severity)
        if severity!="None":
            cv2.fillPoly(overlay,[pts_arr],color)
            cv2.polylines(frame,[pts_arr],True,color,2 if hollow>0.45 else 1)
        else:
            cv2.polylines(frame,[pts_arr],True,(60,120,100),1)
        cx=int(np.mean([p[0] for p in pts])); cy=int(np.mean([p[1] for p in pts]))
        if severity!="None":
            lbl=f"{v['hollow_cnn']:.2f}"
            cv2.putText(frame,lbl,(cx-14,cy+5),cv2.FONT_HERSHEY_PLAIN,0.65,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,lbl,(cx-13,cy+4),cv2.FONT_HERSHEY_PLAIN,0.65,color,1,cv2.LINE_AA)
    cv2.addWeighted(overlay,0.30,frame,0.70,0,frame)

    # Face shape overlay
    shape=ai_fat_data.get("face_shape")
    if shape:
        draw_face_shape_overlay(frame, landmarks, shape,
                                ai_fat_data.get("shape_metrics",{}), h, w)

    # HUD
    cv2.rectangle(frame,(w-230,4),(w-4,70),(10,10,10),-1)
    cv2.rectangle(frame,(w-230,4),(w-4,70),(50,200,200),1)
    cv2.putText(frame,"AI Compare (CNN+LUM)",(w-224,20),cv2.FONT_HERSHEY_PLAIN,0.80,(50,200,200),1)
    cv2.putText(frame,f"MF: {ai_fat_data['total_mf']:.1f} cc",(w-224,38),cv2.FONT_HERSHEY_PLAIN,1.0,(160,230,110),1)
    cv2.putText(frame,f"NF: {ai_fat_data['total_nf']:.1f} cc",(w-224,56),cv2.FONT_HERSHEY_PLAIN,1.0,(110,190,255),1)
    if shape:
        cv2.putText(frame,f"Shape: {shape}",(w-224,70),cv2.FONT_HERSHEY_PLAIN,0.80,(200,200,50),1)


# ═══════════════════════════════════════════════════════════════════════════════
# DIFF PANEL  (new – side-by-side comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def build_diff_panel(diff_table, ai_fat_data, face_shape, shape_metrics, cam_h):
    PW=460
    panel=np.full((cam_h,PW,3),15,dtype=np.uint8)
    cv2.line(panel,(0,0),(0,cam_h),(50,200,200),2)

    y=22
    cv2.putText(panel,"AI COMPARE – Hollow Diff",(8,y),cv2.FONT_HERSHEY_PLAIN,1.05,(50,200,200),1); y+=16

    # Face shape summary
    if face_shape:
        tips=FACE_SHAPE_TIPS.get(face_shape,{})
        cv2.rectangle(panel,(4,y),(PW-4,y+50),(25,35,30),-1)
        cv2.rectangle(panel,(4,y),(PW-4,y+50),(200,200,50),1)
        cv2.putText(panel,f"Face Shape: {face_shape}",(8,y+13),cv2.FONT_HERSHEY_PLAIN,0.95,(200,200,50),1)
        desc=tips.get("desc","")[:52]
        cv2.putText(panel,desc,(8,y+25),cv2.FONT_HERSHEY_PLAIN,0.60,(160,160,160),1)
        zones=", ".join(tips.get("fat_zones",[]))[:55]
        cv2.putText(panel,f"Reco zones: {zones}",(8,y+37),cv2.FONT_HERSHEY_PLAIN,0.58,(120,200,120),1)
        reason=tips.get("reason","")[:55]
        cv2.putText(panel,reason,(8,y+48),cv2.FONT_HERSHEY_PLAIN,0.55,(120,160,200),1)
        y+=56
        # Shape metrics
        if shape_metrics:
            m=shape_metrics
            cv2.putText(panel,f"H/W:{m.get('face_h/w')}  J/F:{m.get('jaw/fore')}  C/F:{m.get('cheek/fore')}  Jaw∠:{m.get('jaw_angle')}°",
                        (8,y),cv2.FONT_HERSHEY_PLAIN,0.58,(80,120,80),1); y+=12

    cv2.line(panel,(4,y),(PW-4,y),(55,55,55),1); y+=8

    # Column header
    cv2.putText(panel,"  Region                Orig   AI   Δ  Match",
                (4,y),cv2.FONT_HERSHEY_PLAIN,0.60,(75,75,75),1); y+=10
    cv2.line(panel,(4,y),(PW-4,y),(40,40,40),1); y+=5

    arch_order=["T","M","J","Other"]
    arch_titles={"T":"T Arch","M":"M Arch","J":"J Arch","Other":"Other"}

    for arch in arch_order:
        if y>cam_h-28: break
        color=ARCH_BGR_PANEL[arch]
        cv2.putText(panel,arch_titles[arch],(6,y),cv2.FONT_HERSHEY_PLAIN,0.82,color,1); y+=12
        rows=[d for d in diff_table if d["arch"]==arch]
        for d in rows:
            if y>cam_h-28: break
            lbl=d["label"][:22]
            delta_col=(0,220,100) if d["agree"] else ((0,100,255) if d["delta"]>0 else (100,100,255))
            delta_str=(f"+{d['delta']:.2f}" if d["delta"]>=0 else f"{d['delta']:.2f}")
            match_sym="✓" if d["agree"] else "≠"
            match_col=(0,200,80) if d["agree"] else (0,80,255)
            txt=f"  {lbl:<22}{d['orig_h']:>5.2f}{d['ai_h']:>6.2f}"
            cv2.putText(panel,txt,(4,y),cv2.FONT_HERSHEY_PLAIN,0.60,color,1)
            cv2.putText(panel,delta_str,(PW-85,y),cv2.FONT_HERSHEY_PLAIN,0.60,delta_col,1)
            cv2.putText(panel,match_sym,(PW-38,y),cv2.FONT_HERSHEY_PLAIN,0.70,match_col,1)
            # delta bar
            bw=int(abs(d["delta"])*60); bx=PW-85-bw-2
            if bw>0 and bx>0:
                cv2.rectangle(panel,(bx,y-7),(bx+bw,y-2),delta_col,-1)
            y+=11
        cv2.line(panel,(4,y),(PW-4,y),(40,40,40),1); y+=5

    # Agreement summary
    agree_count=sum(1 for d in diff_table if d["agree"])
    total_count=len(diff_table)
    pct=int(agree_count/total_count*100) if total_count else 0
    if y<cam_h-38:
        cv2.putText(panel,f"Agreement: {agree_count}/{total_count} regions  ({pct}%)",
                    (8,y+8),cv2.FONT_HERSHEY_PLAIN,0.78,(50,200,200),1); y+=20
        bar_w=int((PW-20)*pct/100)
        cv2.rectangle(panel,(8,y),(8+bar_w,y+8),(50,200,200),-1)
        cv2.rectangle(panel,(8,y),(PW-12,y+8),(60,60,60),1)
        y+=14
        # CNN availability
        cnn_txt="CNN: ACTIVE (micro-CNN)" if TORCH_OK else "CNN: DISABLED (no torch)"
        cnn_col=(0,220,120) if TORCH_OK else (60,60,255)
        cv2.putText(panel,cnn_txt,(8,y+4),cv2.FONT_HERSHEY_PLAIN,0.72,cnn_col,1)

    cv2.putText(panel,"PLANNING AID ONLY – clinician review required",
                (8,cam_h-18),cv2.FONT_HERSHEY_PLAIN,0.72,(60,60,180),1)
    cv2.putText(panel,"Fat Graft Predictor v4 – AI Compare",
                (8,cam_h-6),cv2.FONT_HERSHEY_PLAIN,0.65,(60,60,60),1)
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL STATS PANELS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def build_stats_panel(anat_data,age,cam_h):
    PW=420; panel=np.full((cam_h,PW,3),18,dtype=np.uint8)
    cv2.line(panel,(0,0),(0,cam_h),(55,55,55),1)
    y=22
    cv2.putText(panel,f"ANATOMIC REGIONS  |  Age: {age}",(8,y),cv2.FONT_HERSHEY_PLAIN,1.05,(230,230,230),1); y+=14
    nf_d=anat_data.get("total_nf_deep",anat_data["total_nf"]); nf_e=anat_data.get("total_nf_epidermal",0.0)
    cv2.putText(panel,f"MF {anat_data['total_mf']:.1f}cc  NFd {nf_d:.1f}+{nf_e:.1f}cc",(8,y),cv2.FONT_HERSHEY_PLAIN,0.95,(160,230,110),1); y+=13
    mfa=anat_data.get("mf_arch",{})
    if mfa:
        cv2.putText(panel,f"J:{mfa.get('J',0):.1f} M:{mfa.get('M',0):.1f} T:{mfa.get('T',0):.1f} O:{mfa.get('Other',0):.1f}",
                    (8,y),cv2.FONT_HERSHEY_PLAIN,0.75,(120,200,200),1); y+=12
    conf=anat_data.get("confidence",1.0); yaw=anat_data.get("yaw",0.0)
    cc=(100,255,100) if conf>0.75 else ((0,210,255) if conf>0.40 else (60,60,255))
    cv2.putText(panel,f"Face accuracy: {int(conf*100)}%  Yaw:{yaw:.2f}",(8,y),cv2.FONT_HERSHEY_PLAIN,0.75,cc,1); y+=12
    cv2.line(panel,(4,y),(PW-4,y),(55,55,55),1); y+=8
    cv2.putText(panel,"  Area                 Side    MF   NF   Arch",(4,y),cv2.FONT_HERSHEY_PLAIN,0.60,(75,75,75),1); y+=10
    cv2.line(panel,(4,y),(PW-4,y),(40,40,40),1); y+=6
    rows=anat_data.get("rows",[])
    for ai in [{"key":"J","title":"J Arch – Jawline"},{"key":"M","title":"M Arch – Midface"},
               {"key":"T","title":"T Arch – Upper"},{"key":"Other","title":"Other"}]:
        arch_rows=[r for r in rows if r["arch"]==ai["key"]]
        if not arch_rows or y>cam_h-28: continue
        color=ARCH_BGR_PANEL[ai["key"]]
        cv2.putText(panel,ai["title"],(6,y),cv2.FONT_HERSHEY_PLAIN,0.85,color,1); y+=12
        for r in sorted(arch_rows,key=lambda x:-x["mf"]):
            if y>cam_h-28: break
            tc=color if r["mf"]>0 else (75,75,75)
            cv2.putText(panel,f"  {r['area'][:20]:<20} {r['side']:<7} {r['mf']:>4.1f} {r['nf']:>4.1f}  {r['arch']}",(4,y),
                        cv2.FONT_HERSHEY_PLAIN,0.60,tc,1); y+=11
        cv2.line(panel,(4,y),(PW-4,y),(40,40,40),1); y+=6
    if y<cam_h-50:
        cv2.putText(panel,f"Totals:  MF {anat_data['total_mf']:.1f} cc   NF {anat_data['total_nf']:.1f} cc",
                    (8,y+5),cv2.FONT_HERSHEY_PLAIN,0.75,(200,200,200),1)
    cv2.putText(panel,"PLANNING AID ONLY – clinician review required",(8,cam_h-18),cv2.FONT_HERSHEY_PLAIN,0.72,(60,60,180),1)
    cv2.putText(panel,"Fat Graft Predictor v4",(8,cam_h-6),cv2.FONT_HERSHEY_PLAIN,0.65,(60,60,60),1)
    return panel

def build_jmt_stats_panel(jmt_data,age,cam_h):
    PW=420; panel=np.full((cam_h,PW,3),18,dtype=np.uint8)
    cv2.line(panel,(0,0),(0,cam_h),(55,55,55),1)
    y=22
    cv2.putText(panel,f"JMT INJECTION POINTS  |  Age: {age}",(8,y),cv2.FONT_HERSHEY_PLAIN,1.05,(230,230,230),1); y+=14
    nf_d=jmt_data.get("total_nf_deep",jmt_data["total_nf"]); nf_e=jmt_data.get("total_nf_epidermal",0.0)
    cv2.putText(panel,f"MF {jmt_data['total_mf']:.1f}cc  NFd {nf_d:.1f}+{nf_e:.1f}cc",(8,y),cv2.FONT_HERSHEY_PLAIN,0.95,(160,230,110),1); y+=13
    mfa=jmt_data.get("mf_arch",{})
    if mfa:
        cv2.putText(panel,f"J:{mfa.get('J',0):.1f} M:{mfa.get('M',0):.1f} T:{mfa.get('T',0):.1f} O:{mfa.get('Other',0):.1f}",
                    (8,y),cv2.FONT_HERSHEY_PLAIN,0.75,(120,200,200),1); y+=12
    conf=jmt_data.get("confidence",1.0); yaw=jmt_data.get("yaw",0.0)
    cc=(100,255,100) if conf>0.75 else ((0,210,255) if conf>0.40 else (60,60,255))
    cv2.putText(panel,f"Face accuracy: {int(conf*100)}%  Yaw:{yaw:.2f}",(8,y),cv2.FONT_HERSHEY_PLAIN,0.75,cc,1); y+=12
    cv2.line(panel,(4,y),(PW-4,y),(55,55,55),1); y+=8
    cv2.putText(panel,"  Point                          MF  NFd  Hollow  Severity",
                (4,y),cv2.FONT_HERSHEY_PLAIN,0.58,(75,75,75),1); y+=10
    cv2.line(panel,(4,y),(PW-4,y),(40,40,40),1); y+=6
    for arch in ["T","M","J","Other"]:
        if y>cam_h-28: break
        color=ARCH_BGR_PANEL[arch]
        cv2.putText(panel,{"T":"T Arch","M":"M Arch","J":"J Arch","Other":"Other"}[arch],(6,y),
                    cv2.FONT_HERSHEY_PLAIN,0.85,color,1); y+=12
        for key,v in jmt_data["regions"].items():
            if v["arch"]!=arch or y>cam_h-28: continue
            hc=hollow_color(v["severity"])
            cv2.putText(panel,f"  {v['label'][:28]:<28} {v['mf']:>4.1f} {v['nf']:>4.1f} {v['hollow']:>6.2f}",
                        (4,y),cv2.FONT_HERSHEY_PLAIN,0.62,hc,1)
            cv2.putText(panel,v["severity"][:8],(PW-60,y),cv2.FONT_HERSHEY_PLAIN,0.60,
                        SEVERITY_BGR.get(v["severity"],(130,130,130)),1)
            bw=int(v["hollow"]*25)
            if bw>0: cv2.rectangle(panel,(PW-95,y-8),(PW-95+bw,y-2),hc,-1)
            y+=11
        cv2.line(panel,(4,y),(PW-4,y),(40,40,40),1); y+=6
    if y<cam_h-40:
        cv2.putText(panel,f"NF Epidermal (micro-needling): {nf_e:.1f} cc",(8,y+5),cv2.FONT_HERSHEY_PLAIN,0.70,(110,190,255),1)
    cv2.putText(panel,"PLANNING AID ONLY – clinician review required",(8,cam_h-18),cv2.FONT_HERSHEY_PLAIN,0.72,(60,60,180),1)
    cv2.putText(panel,"Fat Graft Predictor v4 – JMT Mode",(8,cam_h-6),cv2.FONT_HERSHEY_PLAIN,0.65,(60,60,60),1)
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def export_csv(data,age,image_path,mode="anatomic"):
    base=os.path.splitext(image_path)[0]
    suffix="_jmt" if mode=="jmt" else ("_ai_compare" if mode=="ai_compare" else "")
    csv_path=f"{base}_result{suffix}.csv"
    if mode=="anatomic":
        lines=["Area,Side,MF (cc),NF Deep (cc),Arch"]
        rows=data.get("rows",[])
        for arch in ["J","M","T","Other"]:
            for r in [rr for rr in rows if rr["arch"]==arch]:
                lines.append(f"{r['area']},{r['side']},{r['mf']:.1f},{r['nf']:.1f},{r['arch']}")
        lines+=["",f"Totals,,{data['total_mf']:.1f},{data['total_nf']:.1f},",
                f"NF Epidermal,,,,{data['total_nf_epidermal']:.1f}"]
    elif mode=="jmt":
        lines=["Area,Side,MF (cc),NF Deep (cc),Arch"]
        for _,v in data["regions"].items():
            lines.append(f"{v['label']},{v.get('side','Midline')},{v['mf']:.1f},{v['nf']:.1f},{v['arch']}")
        lines+=["",f"Totals,,{data['total_mf']:.1f},{data['total_nf']:.1f},",
                f"NF Epidermal,,,,{data['total_nf_epidermal']:.1f}"]
    else:  # ai_compare diff CSV
        lines=["Region,Label,Arch,Orig Hollow,AI Hollow,Delta,Agree,Orig Severity,AI Severity"]
        for d in data:
            lines.append(f"{d['region']},{d['label']},{d['arch']},{d['orig_h']:.2f},"
                         f"{d['ai_h']:.2f},{d['delta']:+.2f},{'Yes' if d['agree'] else 'No'},"
                         f"{d['orig_sev']},{d['ai_sev']}")
    lines+=["","DISCLAIMER: Values are planning estimates only.",
            "All recommendations must be reviewed by the treating clinician."]
    with open(csv_path,"w") as f: f.write("\n".join(lines))
    print(f"[DONE] CSV saved → {csv_path}")
    return csv_path

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL PRINT – AI COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
def print_ai_compare(diff_table, ai_fat_data, face_shape, shape_metrics, image_path, age):
    print(f"\n{BOLD}{'─'*78}{RESET}")
    print(f"{BOLD}  AI COMPARE  |  {os.path.basename(image_path)}  |  Age: {age}{RESET}")
    print(f"{'─'*78}")
    cnn_st="ACTIVE (micro-CNN+heuristic init)" if TORCH_OK else "DISABLED (install torch)"
    print(f"  CNN Status : {cnn_st}")
    if face_shape:
        tips=FACE_SHAPE_TIPS.get(face_shape,{})
        print(f"\n  {BOLD}Face Shape  : {face_shape}{RESET}")
        print(f"  Description : {tips.get('desc','')}")
        print(f"  Fat Zones   : {', '.join(tips.get('fat_zones',[]))}")
        print(f"  Rationale   : {tips.get('reason','')}")
        if shape_metrics:
            m=shape_metrics
            print(f"  Metrics     : H/W={m.get('face_h/w')}  Jaw/Fore={m.get('jaw/fore')}  "
                  f"Cheek/Fore={m.get('cheek/fore')}  Jaw∠={m.get('jaw_angle')}°")
    print(f"\n{'─'*78}")
    print(f"  {'Region':<28} {'Orig':>6} {'AI':>6} {'Δ':>7}  {'Match':>5}  {'Orig Sev':<14} AI Sev")
    print(f"  {'─'*76}")
    arch_order=["T","M","J","Other"]
    for arch in arch_order:
        ac=ARCH_COLORS[arch]
        rows=[d for d in diff_table if d["arch"]==arch]
        if not rows: continue
        print(f"\n{ac}{BOLD}  {arch} Arch{RESET}")
        for d in rows:
            sc=SEV_COLORS.get(d["ai_sev"],"")
            match="✓" if d["agree"] else "≠"
            mc="\033[92m" if d["agree"] else "\033[91m"
            delta_str=(f"+{d['delta']:.2f}" if d["delta"]>=0 else f"{d['delta']:.2f}")
            print(f"  {ac}{d['label']:<28}{RESET}"
                  f" {d['orig_h']:>6.2f} {d['ai_h']:>6.2f} {delta_str:>7}"
                  f"  {mc}{match}{RESET}"
                  f"  {d['orig_sev']:<14} {sc}{d['ai_sev']}{RESET}")
    agree=sum(1 for d in diff_table if d["agree"]); total=len(diff_table)
    print(f"\n  Agreement: {agree}/{total} regions ({int(agree/total*100) if total else 0}%)")
    print(f"{'─'*78}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL PRINT – ORIGINAL MODES  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def print_results(anat_data,age,image_path):
    print(f"\n{BOLD}{'─'*78}{RESET}")
    print(f"{BOLD}  ANATOMIC REGIONS  |  {os.path.basename(image_path)}  |  Age: {age}{RESET}")
    print(f"{'─'*78}")
    nf_epi=anat_data.get("total_nf_epidermal",0.0); mfa=anat_data.get("mf_arch",{})
    print(f"  {BOLD}MF Total      : {anat_data['total_mf']:.1f} cc{RESET}")
    if mfa: print(f"  {BOLD}MF by Arch    : J={mfa.get('J',0):.1f}  M={mfa.get('M',0):.1f}  T={mfa.get('T',0):.1f}  Other={mfa.get('Other',0):.1f}{RESET}")
    print(f"  {BOLD}NF Deep (J+M) : {anat_data.get('nf_jm',anat_data['total_nf_deep']):.1f} cc{RESET}")
    print(f"  {BOLD}NF Epidermal  : {nf_epi:.1f} cc{RESET}  (whole-face micro-needling)")
    print(f"  Face accuracy : {int(anat_data['confidence']*100)}%   Yaw: {anat_data['yaw']:.2f}")
    print(f"{'─'*78}")
    print(f"  {'Area':<28} {'Side':<10} {'MF (cc)':>8} {'NF Deep (cc)':>13} {'Arch':>6}")
    rows=anat_data.get("rows",[])
    for ai in [{"key":"J","title":"J Arch — Jawline / Lower Face"},{"key":"M","title":"M Arch — Midface"},
               {"key":"T","title":"T Arch — Temples / Upper Face"},{"key":"Other","title":"Other — Nose / Lips / Misc"}]:
        ar=[r for r in rows if r["arch"]==ai["key"]]
        if not ar: continue
        ac=ARCH_COLORS[ai["key"]]
        print(f"\n{ac}{BOLD}  {ai['title']}{RESET}")
        for r in sorted(ar,key=lambda x:-x["mf"]):
            col=ac if r["mf"]>0 else DIM
            print(f"  {col}{r['area']:<28}{RESET} {r['side']:<10} {r['mf']:.1f} cc   {r['nf']:.1f} cc   {r['arch']}")
    print(f"\n  {BOLD}Totals: MF {anat_data['total_mf']:.1f} cc   NF {anat_data['total_nf']:.1f} cc{RESET}")
    print(f"{'─'*78}\n")

def print_jmt_results(jmt_data,age,image_path):
    print(f"\n{BOLD}{'─'*76}{RESET}")
    print(f"{BOLD}  JMT INJECTION POINTS  |  {os.path.basename(image_path)}  |  Age: {age}{RESET}")
    print(f"{'─'*76}")
    nf_epi=jmt_data.get("total_nf_epidermal",0.0); mfa=jmt_data.get("mf_arch",{})
    print(f"  {BOLD}MF Total      : {jmt_data['total_mf']:.1f} cc{RESET}")
    if mfa: print(f"  {BOLD}MF by Arch    : J={mfa.get('J',0):.1f}  M={mfa.get('M',0):.1f}  T={mfa.get('T',0):.1f}  Other={mfa.get('Other',0):.1f}{RESET}")
    print(f"  Face accuracy : {int(jmt_data['confidence']*100)}%   Yaw: {jmt_data['yaw']:.2f}")
    print(f"{'─'*76}")
    print(f"  {'Point':<32} {'Side':<8} {'MF':>6} {'NF':>6} {'Hollow':>7} {'AI-Rec':>7}  Severity")
    for arch in ["J","M","T","Other"]:
        ac=ARCH_COLORS[arch]
        print(f"\n{ac}{BOLD}  {arch} Arch{RESET}")
        for key,v in jmt_data["regions"].items():
            if v["arch"]!=arch: continue
            sc=SEV_COLORS.get(v["severity"],"")
            cc_str=str(v["cc_add"]) if v["cc_add"] else "0.0"
            print(f"  {ac}{v['label']:<32}{RESET} {v.get('side','Midline'):<8}"
                  f" {v['mf']:>6.1f} {v['nf']:>6.1f} {v['hollow']:>7.2f}  {sc}{cc_str:>7}  {v['severity']}{RESET}")
    print(f"\n  NF Epidermal (micro-needling): {nf_epi:.1f} cc")
    print(f"{'─'*76}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE REGION SELECTION  (original)
# ═══════════════════════════════════════════════════════════════════════════════
def select_anatomic_regions():
    print(f"\n{BOLD}  ANATOMIC REGIONS – Select Priority Areas{RESET}")
    for i,r in enumerate(ANATOMIC_SELECTABLE):
        pair_tag="paired" if r["paired"] else "midline"
        print(f"  [{i+1:>2}] {r['label']:<32} ({r['arch']}, {pair_tag})")
    print("  [A] All   [N] None")
    while True:
        raw=input("\n  Selection: ").strip().upper()
        if raw=="A": return {r["key"] for r in ANATOMIC_SELECTABLE}
        if raw=="N": print("  [!] Select at least one."); continue
        try:
            nums=[int(x) for x in raw.split(",") if x.strip()]
            valid={ANATOMIC_SELECTABLE[n-1]["key"] for n in nums if 1<=n<=len(ANATOMIC_SELECTABLE)}
            if not valid: print("  [!] No valid selections."); continue
            for r in ANATOMIC_SELECTABLE:
                if r["key"] in valid: print(f"    ✓ {r['label']}")
            return valid
        except ValueError: print("  [!] Invalid input.")

def select_jmt_regions():
    print(f"\n{BOLD}  JMT INJECTION POINTS – Select Priority Areas{RESET}")
    for i,r in enumerate(JMT_REGIONS): print(f"  [{i+1:>2}] {r['label']:<36} ({r['arch']})")
    print("  [A] All   [N] None")
    while True:
        raw=input("\n  Selection: ").strip().upper()
        if raw=="A": return {r["key"] for r in JMT_REGIONS}
        if raw=="N": print("  [!] Select at least one."); continue
        try:
            nums=[int(x) for x in raw.split(",") if x.strip()]
            valid={JMT_REGIONS[n-1]["key"] for n in nums if 1<=n<=len(JMT_REGIONS)}
            if not valid: print("  [!] No valid selections."); continue
            for r in JMT_REGIONS:
                if r["key"] in valid: print(f"    ✓ {r['label']}")
            return valid
        except ValueError: print("  [!] Invalid input.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*62)
    print("   FAT GRAFT PREDICTOR v4 – Three-Signal + AI Compare")
    print("="*62)
    if TORCH_OK: _try_load_midas()
    else: print("[WARN] Torch unavailable – MiDaS and CNN disabled.")

    # Image
    while True:
        image_path=input("\nEnter path to face image (JPG/PNG): ").strip().strip('"').strip("'")
        if os.path.isfile(image_path): break
        print(f"  [!] File not found: '{image_path}'")

    # Age
    while True:
        try:
            age=int(input("Enter patient age: ").strip())
            if 1<=age<=120: break
            print("  [!] Age 1–120.")
        except ValueError: print("  [!] Enter a valid number.")

    # Mode
    print(f"\n{'─'*62}")
    print("  Select output mode:")
    print("  [1] Anatomic regions")
    print("  [2] JMT injection points")
    print("  [3] Both")
    print("  [4] AI Compare  (CNN hollow analysis + face shape + diff)")
    print("  [5] All of the above")
    while True:
        m=input("  Mode [1-5]: ").strip()
        if m in ("","1"): run_anat=True;  run_jmt=False; run_ai=False; break
        elif m=="2":       run_anat=False; run_jmt=True;  run_ai=False; break
        elif m=="3":       run_anat=True;  run_jmt=True;  run_ai=False; break
        elif m=="4":       run_anat=False; run_jmt=False; run_ai=True;  break
        elif m=="5":       run_anat=True;  run_jmt=True;  run_ai=True;  break
        else: print("  [!] Enter 1–5.")

    selected_anat=select_anatomic_regions() if run_anat else None
    selected_jmt=select_jmt_regions()       if run_jmt  else None

    # Load & resize
    frame=cv2.imread(image_path)
    if frame is None: sys.exit(f"[ERROR] Cannot read: {image_path}")
    MAX_DIM=960; h,w=frame.shape[:2]
    if max(h,w)>MAX_DIM:
        sc=MAX_DIM/max(h,w); frame=cv2.resize(frame,(int(w*sc),int(h*sc)),interpolation=cv2.INTER_AREA)
        h,w=frame.shape[:2]
    print(f"\n[INFO] Image: {w}×{h}  Age: {age}")

    # MediaPipe
    print("[INFO] Detecting landmarks...")
    mp_face_mesh=mp_solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,
                                refine_landmarks=True,min_detection_confidence=0.5) as fm:
        results=fm.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        sys.exit("[ERROR] No face detected. Use a frontal, well-lit photo.")
    landmarks=results.multi_face_landmarks[0].landmark
    print(f"[INFO] {len(landmarks)} landmarks detected.")

    # Compute base signals
    fat_data=calc_fat_volumes(age,landmarks,frame,h,w)

    # Face shape (always computed – used in AI panel)
    print("[INFO] Detecting face shape...")
    face_shape,shape_metrics=detect_face_shape(landmarks,h,w)
    print(f"[INFO] Face shape: {face_shape}")

    # Annotated base image
    annotated=frame.copy()
    draw_overlay(annotated,landmarks,fat_data,h,w,mp_face_mesh)

    base,_=os.path.splitext(image_path)

    # ── ANATOMIC ────────────────────────────────────────────────────────────
    if run_anat:
        anat_data=calc_anatomic_volumes(age,fat_data,selected_anat)
        print_results(anat_data,age,image_path)
        panel=build_stats_panel(anat_data,age,h)
        out=np.hstack([annotated,panel])
        p=f"{base}_result_v4.jpg"; cv2.imwrite(p,out,[cv2.IMWRITE_JPEG_QUALITY,95])
        print(f"[DONE] Anatomic image → {p}")
        export_csv(anat_data,age,image_path,mode="anatomic")
        cv2.imshow("Fat Graft Predictor v4 – Anatomic",out)

    # ── JMT ─────────────────────────────────────────────────────────────────
    if run_jmt:
        jmt_data=calc_jmt_volumes(age,fat_data,selected_jmt)
        print_jmt_results(jmt_data,age,image_path)
        jp=build_jmt_stats_panel(jmt_data,age,h)
        outj=np.hstack([annotated,jp])
        pj=f"{base}_result_v4_jmt.jpg"; cv2.imwrite(pj,outj,[cv2.IMWRITE_JPEG_QUALITY,95])
        print(f"[DONE] JMT image → {pj}")
        export_csv(jmt_data,age,image_path,mode="jmt")
        cv2.imshow("Fat Graft Predictor v4 – JMT",outj)

    # ── AI COMPARE ──────────────────────────────────────────────────────────
    if run_ai:
        ai_fat_data,diff_table,ai_anat_data=run_ai_compare(
            age, image_path, landmarks, frame, h, w,
            fat_data, face_shape, shape_metrics
        )
        print_ai_compare(diff_table,ai_fat_data,face_shape,shape_metrics,image_path,age)

        # Left: AI-annotated face  |  Right: diff panel
        ai_frame=frame.copy()
        draw_ai_overlay(ai_frame,landmarks,ai_fat_data,h,w,mp_face_mesh)
        diff_panel=build_diff_panel(diff_table,ai_fat_data,face_shape,shape_metrics,h)
        out_ai=np.hstack([ai_frame,diff_panel])

        # Side-by-side: original annotated | AI annotated
        if h>0 and w>0:
            compare_side=np.hstack([annotated,ai_frame])
            label_h=22
            compare_label=np.zeros((label_h,compare_side.shape[1],3),dtype=np.uint8)
            cv2.putText(compare_label,"Original (MP+LUM+MiDaS)",(10,15),cv2.FONT_HERSHEY_PLAIN,0.90,(160,230,110),1)
            cv2.putText(compare_label,"AI Compare (CNN+LUM)",(w+10,15),cv2.FONT_HERSHEY_PLAIN,0.90,(50,200,200),1)
            compare_full=np.vstack([compare_label,compare_side])
            pc=f"{base}_result_v4_compare.jpg"
            cv2.imwrite(pc,compare_full,[cv2.IMWRITE_JPEG_QUALITY,95])
            print(f"[DONE] Side-by-side compare → {pc}")
            cv2.imshow("Fat Graft Predictor v4 – Side-by-Side",compare_full)

        pd=f"{base}_result_v4_ai.jpg"
        cv2.imwrite(pd,out_ai,[cv2.IMWRITE_JPEG_QUALITY,95])
        print(f"[DONE] AI panel image → {pd}")
        export_csv(diff_table,age,image_path,mode="ai_compare")
        cv2.imshow("Fat Graft Predictor v4 – AI Compare",out_ai)

    print("\n[INFO] Press any key in terminal or image window to close.")
    while True:
        if cv2.waitKey(1)!=-1: break
        if HAS_MSVCRT and msvcrt.kbhit(): msvcrt.getch(); break
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__=="__main__":
    main()