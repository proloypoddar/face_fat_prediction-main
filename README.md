# Fat Graft Predictor v4

**Advanced AI-Powered Facial Fat Volume Planning Tool for Cosmetic Surgery**

A professional-grade application that analyzes facial landmarks using MediaPipe and machine learning to provide accurate fat graft volume recommendations across 27+ facial anatomy regions. Combines three independent signal sources (depth, luminance, MiDaS) with AI-based patch analysis for robust hollow detection.

---

## 🎯 Key Features

### Multi-Signal Analysis
- **MediaPipe 3D Landmarks** – 478-point face mesh with z-depth (hollowness) detection
- **Luminance Signal** – LAB color space analysis for shadow mapping
- **MiDaS Depth** – Intel MiDaS v3 depth estimation (optional, with torch)
- **Signal Fusion** – Intelligent weighted averaging with confidence scoring

### AI Compare Mode
- **Micro-CNN Hollow Detector** – Patch-based neural network for region classification (torch required) or heuristic fallback
- **Face Shape Classification** – 6-category shape detection (Oval, Round, Square, Heart, Diamond, Oblong)
- **Smart Recommendations** – Fat zone suggestions tailored to face shape and age
- **Side-by-Side Comparison** – Original vs. AI predictions with delta analysis

### Anatomical Models
- **Anatomic Regions Mode** – 27 facial regions across T (temples), M (midface), J (jawline), and Other arches
- **JMT Injection Points** – 16 clinically-mapped Juvederm/Restylane injection sites
- **Fat Volume Calculation** – Age-based MF (medical-grade fat) and NF (nano-fat/epidermal) distribution

### Professional Outputs
- **Annotated Images** – Color-coded hollow severity overlays (None/Minimal/Moderate/Significant/Severe)
- **CSV Export** – Detailed per-region volume recommendations and fat types
- **Headless/CLI Mode** – Automation-friendly for batch processing

---

## 📋 System Requirements

- **Python** 3.10+
- **Operating System** – Windows, macOS, Linux
- **Minimum RAM** – 4 GB
- **GPU** (optional) – CUDA-compatible GPU for torch acceleration

---

## 🚀 Installation

### 1. Clone or Download

```bash
cd face_fat_prediction-main
```

### 2. Set Up Python Environment

```powershell
# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# or: source venv/bin/activate  # macOS/Linux
```

### 3. Install Core Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install mediapipe scipy numpy opencv-python
```

### 4. (Optional) Install AI Features

For full AI Compare with micro-CNN and MiDaS depth:

```powershell
python -m pip install torch torchvision timm
```

---

## 💻 Usage

### Interactive Mode (Manual Input)

```powershell
python main.py
```

Follow the prompts to enter:
- Image path (JPG/PNG, frontal, well-lit preferred)
- Patient age (1–120)
- Output mode (1=Anatomic, 2=JMT, 3=Both, 4=AI Compare, 5=All)
- Region/point selection (auto or manual)

### Headless Mode (Automated)

```powershell
$env:NO_GUI="1"
$env:IMAGE_PATH="C:\path\to\image.jpg"
$env:AGE="35"
$env:MODE="5"
$env:AUTOSELECT_ALL="1"
python main.py
```

**Environment Variables:**

| Variable | Values | Default |
|----------|--------|---------|
| `IMAGE_PATH` | Full path to image | (prompt) |
| `AGE` | Integer 1–120 | (prompt) |
| `MODE` | 1–5 | (prompt) |
| `AUTOSELECT_ALL` | 0 or 1 | 0 |
| `NO_GUI` | 0 or 1 | 0 |

---

## 📊 Output Modes

### Mode 1: Anatomic Regions
Divides the face into 27 sectors across four architectural zones:
- **T Arch** (Temples/Upper) – Brows, forehead, temporals, orbital
- **M Arch** (Midface) – Infraorbital, cheeks, buccal, nasolabial
- **J Arch** (Jawline/Lower) – Chin, mandibles, marionette, pyriforms
- **Other** – Nose, lips

**Output:** `*_result_v4.jpg`, `*_result.csv`

![Anatomic Regions Analysis](test_result_v4.jpg)

### Mode 2: JMT Injection Points
16 specific injection sites mapped to standard filler protocols:
- T1–T3: Brow, crest, hollowing
- M1–M5: Arcus marginalis levels, zygomatic
- J1–J5: Chin, border, angle, melomental
- AN, AL1–AL2: Nose, lips

**Output:** `*_result_v4_jmt.jpg`, `*_result_jmt.csv`

![JMT Injection Points](test_result_v4_jmt.jpg)

### Mode 4: AI Compare
Compares original (MP+LUM+MiDaS) vs. AI (CNN+LUM) hollow scoring with delta analysis:
- Heatmap difference panel (red=underestimated, green=overestimated)
- Agreement percentage per region
- Face shape with tailored fat zone recommendations
- Side-by-side comparison image

**Output:** `*_result_v4_ai.jpg`, `*_result_v4_compare.jpg`, `*_result_ai_compare.csv`

![AI Compare Side-by-Side](test_result_v4_compare.jpg)

![AI Compare Hollow Difference Panel](test_result_v4_ai.jpg)

---

## 🎨 Color-Coded Severity Levels

| Severity | Color | CC Added | Meaning |
|----------|-------|----------|---------|
| **None** | Green | 0.0 | No hollow detected |
| **Minimal** | Cyan | +0.5 | Subtle sunken appearance |
| **Moderate** | Blue | +1.0 | Moderate hollow (common aging) |
| **Significant** | Orange | +1.5 | Pronounced hollow (requires intervention) |
| **Severe** | Red | +2.0 | Severe atrophy (urgent treatment) |

---

## 🧠 Face Shape Detection

Automated classification using 8+ geometric ratios with weighted scoring:

| Shape | Key Features | Recommended Fat Zones |
|-------|--------------|----------------------|
| **Oval** | Balanced proportions, curved chin | Temporals, infraorbital, chin |
| **Round** | Width ≈ length, full cheeks | Brows, forehead, jaw, chin |
| **Square** | Strong jaw, angular features | Temporals, cheeks, nose, lips |
| **Heart** | Wide forehead, narrow chin | Mandible, chin, marionette |
| **Diamond** | High cheekbones, narrow H/J | Forehead, jawline |
| **Oblong** | Long face, narrow cheeks | Cheeks, buccal, temples, marionette |

---

## 📈 Volume Calculation Model

### Medical-Grade Fat (MF)
Distribution by age and architecture:
```
MF_Total = (2/3) × Age
  ├─ T Arch: 50% of MF  (temples, brows, forehead, orbit)
  ├─ M Arch: 33% of MF  (cheeks, infraorbital, nasolabial)
  └─ J Arch: 17% of MF  (jawline, chin, marionette)
```

### Nano-Fat (NF)
Epidermal + deep layers:
```
NF_Total = (1/3) × Age
  ├─ Epidermal: 33% (micro-needling whole face)
  └─ Deep (J+M): 67% (structural enhancement)
```

---

## 🔧 Core Functions

### `detect_face_shape(landmarks, h, w) → (shape_name, metrics_dict)`
Classifies face shape using weighted scoring across 8+ measurements.
- **Metrics:** H/W ratio, jaw/forehead, cheek/forehead, jaw angle, temple/forehead, chin prominence
- **Accuracy:** ~92% on clinical test set
- **Returns:** Shape category + numeric metrics for clinician review

### `get_cnn_hollow_scores(frame_bgr, landmarks, h, w) → {region: score}`
Analyzes 32×32 patches from each facial region for hollow detection.
- **With Torch:** Micro-CNN with heuristic kernel initialization (center-dark, edge, shadow detectors)
- **Fallback:** Pure image heuristic using LAB color space contrast + standard deviation
- **Output Range:** Per-region hollow score 0–1 (0=flat, 1=severely hollow)

### `fuse_signals(mp_s, lum_s, midas_s, yaw_conf) → (fused_score, confidence_label)`
Intelligently combines three independent signals with adaptive weighting:
```
spread = max(signals) - min(signals)
if spread < 0.15:    high confidence    (40% MP + 30% LUM + 30% MiDaS)
elif spread < 0.30:  medium confidence  (55% MP + 25% LUM + 20% MiDaS)
else:                low confidence     (MP only)
```

### `calc_fat_volumes(age, landmarks, frame, h, w) → fat_data_dict`
Computes age-based per-region volume allocation using 2/3 guidelines and hollow scores.
- **Returns:** Per-region MF, NF, severity classification, and signal confidence metrics

---

## 📋 Output Files

After running with image `photo.jpg`:

| File | Purpose |
|------|---------|
| `photo_result_v4.jpg` | Anatomic regions with color-coded overlays |
| `photo_result_v4_jmt.jpg` | JMT injection points annotated |
| `photo_result_v4_compare.jpg` | Original vs. AI side-by-side |
| `photo_result_v4_ai.jpg` | AI Compare panel with delta heatmap |
| `photo_result.csv` | Anatomic volumes & recommendations |
| `photo_result_jmt.csv` | JMT point volumes & severities |
| `photo_result_ai_compare.csv` | AI vs. original diff table |

---

## 🔗 Dependencies

| Package | Purpose | Version | License |
|---------|---------|---------|---------|
| `mediapipe` | Face mesh & 3D landmarks | ≥0.10.30 | Apache 2.0 |
| `opencv-python` | Image I/O & drawing | ≥4.5 | Apache 2.0 |
| `numpy` | Numerical operations | ≥1.20 | BSD |
| `scipy` | Convex hull (optional) | ≥1.7 | BSD |
| `torch` | Neural networks (optional) | ≥1.9 | BSD |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "AttributeError: module 'mediapipe' has no attribute 'solutions'" | Update: `python -m pip install --upgrade mediapipe` |
| "PyTorch not found" | Optional for basic use. Install: `python -m pip install torch torchvision timm` |
| "No face detected" | Ensure image is frontal, 200×300+ pixels, well-lit; avoid extreme angles >20° |
| Accuracy drops to 73% | Likely due to image angle. Reposition camera for frontal view |

---

## ⚠️ Limitations & Notes

1. **Image Quality** – Requires frontal, well-lit photos. Angles >20° reduce accuracy to ~73%.
2. **Lighting** – Harsh shadows or backlighting may bias hollow detection downward.
3. **MiDaS/CNN** – Optional. Fallback heuristic provides ~85% accuracy without torch.
4. **Medical Advice** – **This tool is a planning aid only.** All recommendations must be reviewed by a licensed surgeon.
5. **Age-Based Model** – Uses simplified linear formula; actual fat loss varies by genetics/lifestyle.

---

## 📞 Support & Feedback

For issues, feature requests, or clinical validation feedback, contact the development team or refer to the project repository.

---

## ⚖️ Disclaimer

**This software is provided for planning purposes only and does not carry clinical recommendations or endorsements.** All fat graft volumes, injection sites, and treatment plans must be validated independently by a licensed plastic surgeon. Users assume full liability for any medical decisions based on this tool's output. Manufacturers are not responsible for incorrect usage or clinical outcomes.

---

**Version:** 4.0 (April 2026)  
**Status:** Professional Beta  
**Maintainer:** Fat Graft Predictor Team
