# 03 — Environment Setup

## Exact Reproduction Steps

Follow these steps **in order** to recreate the working environment from scratch.
All commands are run in a **Windows PowerShell or CMD** terminal.

---

## Step 1 — Python Version

Install **Python 3.14.0** from https://www.python.org/downloads/
(Any Python 3.10+ should work but 3.14.0 is the tested version.)

Verify:
```powershell
python --version
# Expected: Python 3.14.0
```

---

## Step 2 — Project Directory

```powershell
# Navigate to your project root
cd "C:\Users\<YOU>\...\CV_Project"

# Create the application folder
mkdir parking_counter
cd parking_counter

# Create assets subfolder
mkdir assets
```

---

## Step 3 — Create Virtual Environment

```powershell
# Inside parking_counter/
python -m venv .venv

# Activate it
.venv\Scripts\activate
```

After activation your prompt will show `(.venv)`.

**IMPORTANT:** The virtual environment MUST be at `parking_counter\.venv`.
If you create it at the workspace root level, the app will still run but the inner venv
is the one used when launching from `parking_counter/`.

---

## Step 4 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## Step 5 — Install CPU-Only PyTorch FIRST

> **Why first?** If you run `pip install -r requirements.txt` without this step,
> pip may install a CUDA-enabled torch build, which causes `WinError 1114 c10.dll`
> on machines without a compatible NVIDIA GPU driver.

```powershell
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```powershell
python -c "import torch; print(torch.__version__)"
# Expected: 2.2.2+cpu
```

---

## Step 6 — Install All Other Dependencies

```powershell
pip install opencv-python>=4.8.0 numpy>=1.24.0 ultralytics>=8.0.0 PyQt5>=5.15.0 PyYAML>=6.0 scipy>=1.11.0 Pillow>=10.0.0
```

Or use the requirements.txt (which pins the exact CPU-torch versions):

```powershell
pip install -r requirements.txt
```

### Exact `requirements.txt` content:
```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
PyQt5>=5.15.0
PyYAML>=6.0
scipy>=1.11.0
Pillow>=10.0.0
torch==2.2.2+cpu
torchvision==0.17.2+cpu
--extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Step 7 — Verify Installation

```powershell
python -c "
import cv2, numpy, ultralytics, PyQt5, yaml, scipy, PIL, torch, torchvision
print('cv2:', cv2.__version__)
print('numpy:', numpy.__version__)
print('torch:', torch.__version__)
print('All imports OK')
"
```

Expected output (exact versions may vary for non-pinned packages):
```
cv2: 4.13.0
numpy: 2.4.2
torch: 2.2.2+cpu
All imports OK
```

---

## Step 8 — YOLO Weights (Auto-Download)

On first run, ultralytics will automatically download `yolov8n.pt` to `assets/`.
If you want to pre-download it manually:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Copy the downloaded file to parking_counter/assets/yolov8n.pt
```

Or simply run the app — it will download on first detection pass.

---

## Step 9 — Place Source Files

Place all Python source files directly inside `parking_counter/`:
- `main.py`
- `ui.py`
- `pipeline.py`
- `video_handler.py`
- `roi_manager.py`
- `detector.py`
- `tracker.py`
- `classifier.py`
- `draw_rois.py`
- `config.yaml`

(See each module's spec document for exact file contents.)

---

## Step 10 — Run the Application

```powershell
# Make sure .venv is activated and you are inside parking_counter/
cd parking_counter
.venv\Scripts\activate
python main.py
```

To use a non-default config:
```powershell
python main.py --config path\to\other_config.yaml
```

---

## Troubleshooting

### `WinError 1114` or `c10.dll` crash on import
**Cause:** CUDA PyTorch installed on a system without a matching GPU driver.
**Fix:**
```powershell
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu --upgrade
```

### `ModuleNotFoundError` for any package
**Cause:** Wrong virtual environment activated, or requirements not installed.
**Fix:** Ensure you activated `parking_counter\.venv\Scripts\activate` then re-install requirements.

### Two venvs exist
If there is both a `CV_Project\.venv` and a `CV_Project\parking_counter\.venv`,
always activate and use the **inner** one (`parking_counter\.venv`).

### `Cannot open video source`
Ensure the video file path passed to "Open Video" is valid and the codec is supported by OpenCV.
For H.265/HEVC, install the appropriate codec pack or transcode to H.264.

### `cv2.error: Unknown C++ exception` / GaussianBlur crash
This was a bug that has been fixed. If encountered, ensure you have the latest version of
`roi_manager.py` that returns `None` from `extract_patch` on degenerate ROIs, and the latest
`classifier.py` that guards for `None` patches.

---

## Package Version Reference

| Package | Min Version | Tested Version |
|---------|-------------|---------------|
| Python | 3.10 | 3.14.0 |
| opencv-python | 4.8.0 | 4.13.0 |
| numpy | 1.24.0 | 2.4.2 |
| ultralytics | 8.0.0 | latest |
| PyQt5 | 5.15.0 | latest |
| PyYAML | 6.0 | latest |
| scipy | 1.11.0 | 1.17.0 |
| Pillow | 10.0.0 | latest |
| torch | **2.2.2+cpu** | 2.2.2+cpu |
| torchvision | **0.17.2+cpu** | 0.17.2+cpu |
