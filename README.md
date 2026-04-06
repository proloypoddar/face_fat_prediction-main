# Fat Graft Predictor v3

A clinical planning tool for estimating fat graft volumes based on patient age and 2/3 guidelines. 
It utilizes a three-signal fusion approach for high-accuracy facial analysis.

## 🚀 Key Features

- **Three-Signal Fusion Analysis**:
  - **MediaPipe Z-depth & Geometry**: Analyzes facial landmarks for z-variance and convexity.
  - **OpenCV LAB Luminance**: Detects shadows and hollow areas via luminance contrast.
  - **MiDaS Monocular Depth**: Local depth estimation for robust measurement.
- **2/3 Volume Guidelines**: Implements the standardized rule:
  - **Microfat (MF)** = 2/3 × Age (cc)
  - **Nanofat (NF)** = 1/3 × Age (cc)
- **Advanced Mapping**:
  - **JMT Injection Points**: T1-T3, M1-M5, J1-J5, AN, AL1/AL2.
  - **Anatomic Regions**: Brows, Forehead, Temporals, etc., with manual selection.
- **Reporting**: Automated CSV export and annotated image generation.

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/apusaha0011/face_fat_prediction.git
   cd face_fat_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For MiDaS depth support:
   ```bash
   pip install torch torchvision timm
   ```

## 📖 Usage

Run the predictor:
```bash
python 06.py
```

1. **Patient Photo**: Provide a path to a frontal face photo (JPG/PNG).
2. **Age**: Enter the patient's age to calculate total volumes.
3. **Mode selection**: Choose between Anatomic Regions (1), JMT Injection Points (2), or Both (3).
4. **Selection**: Manually select priority areas for volume allocation.
5. **Results**: Review the annotated image and CSV results.

## ⚠️ Disclaimer

**THIS IS NOT A MEDICAL DEVICE.**
This tool is intended as a clinical planning aid only. It provides estimates based on generalized guidelines. All volume recommendations **MUST** be reviewed and adjusted by the treating clinician based on individual patient anatomy, asymmetries, and clinical judgment.

## 📄 License

This project is for planning purposes and does not carry a commercial license. Use at your own risk.
