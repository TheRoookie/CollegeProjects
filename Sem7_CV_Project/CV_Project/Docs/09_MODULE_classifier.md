# 09 — Module: `classifier.py`

**Syllabus Topics:** Image Filtering, Image Classification
**File:** `parking_counter/classifier.py` (267 lines)

---

## Purpose

Binary classifier that determines whether a parking space is **Empty** or **Occupied**.
Provides two interchangeable backends:

1. **`PixelCountClassifier`** — OpenCV pipeline: grayscale → Gaussian blur → adaptive threshold → pixel ratio
2. **`CNNClassifier`** — PyTorch convolutional network (only used when `method="cnn"` explicitly)
3. **`SpaceClassifier`** — Façade that selects the backend based on config

---

## Enum: `SpaceStatus`

```python
class SpaceStatus(IntEnum):
    EMPTY    = 0
    OCCUPIED = 1
```

Used as the return type of all `classify()` methods.

---

## Class: `PixelCountClassifier`

**Implements syllabus topics: Image Filtering + Image Classification**

### Constructor

```python
PixelCountClassifier(
    pixel_threshold: float = 0.18,
    blur_kernel: int = 5,
    adaptive_block: int = 11,
    adaptive_c: int = 2,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pixel_threshold` | 0.18 | Fraction of non-zero pixels above which the spot is "occupied" |
| `blur_kernel` | 5 | Gaussian blur kernel size — must be odd; auto-corrected to odd |
| `adaptive_block` | 11 | adaptiveThreshold block size — must be odd; auto-corrected to odd |
| `adaptive_c` | 2 | Constant subtracted from local mean in adaptive threshold |

**Auto-correction in constructor:**
```python
self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
self.adaptive_block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
```

### Method: `classify(patch: np.ndarray) → SpaceStatus`

Full pipeline with all safety guards:

```python
def classify(self, patch: np.ndarray) -> SpaceStatus:
    # ---- Guard 1: None check ----
    if patch is None or patch.ndim < 2:
        return SpaceStatus.EMPTY

    ph, pw = patch.shape[:2]

    # ---- Guard 2: Zero-dimension check ----
    if ph == 0 or pw == 0:
        return SpaceStatus.EMPTY

    # ---- Step 1: Grayscale (Image Filtering) ----
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # ---- Step 2: Gaussian Blur (Image Filtering) ----
    # Clamp kernel to patch dimensions (CRITICAL: prevents GaussianBlur crash)
    ksize = min(self.blur_kernel, ph, pw)
    ksize = ksize if ksize % 2 == 1 else max(1, ksize - 1)  # must be odd
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # ---- Step 3: Adaptive Thresholding (Image Filtering → binarisation) ----
    # block must be odd and smaller than patch dimensions
    block = min(self.adaptive_block, ph, pw)
    block = block if block % 2 == 1 else max(1, block - 1)

    if block < 3:
        # Fallback for very tiny patches: plain Otsu threshold
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            self.adaptive_c,
        )

    # ---- Step 4: Pixel Ratio Decision (Image Classification) ----
    total_pixels = thresh.size
    filled_pixels = int(np.count_nonzero(thresh))
    ratio = filled_pixels / total_pixels if total_pixels > 0 else 0.0

    return SpaceStatus.OCCUPIED if ratio >= self.pixel_threshold else SpaceStatus.EMPTY
```

**Why all these guards?**
Without them, `cv2.GaussianBlur` or `cv2.adaptiveThreshold` will throw a
`cv2.error: Unknown C++ exception` when the patch has zero or near-zero dimensions.
This crash was encountered in production and fixed by adding all guards above.

### Method: `preprocess_for_display(patch) → np.ndarray`

Returns the thresholded binary image for debug visualisation (not used in main pipeline).

---

## Class: `CNNClassifier`

**Used only when `method="cnn"` in config.yaml. Requires a trained checkpoint.**

### Constructor

```python
CNNClassifier(
    checkpoint_path: str | Path,
    input_size: tuple[int, int] = (64, 64),
    device: str = "cpu",
)
```

### Architecture: `ParkingCNN`

Defined inside `_build_model()`:

```python
class ParkingCNN(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                     # 64×64 → 32×32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                     # 32×32 → 16×16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),                   # Binary output: [empty, occupied]
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Input tensor shape: `(1, 3, 64, 64)` — batch × channels × height × width.

### Method: `load() → None`

```python
model = self._build_model()
state = torch.load(self.checkpoint_path, map_location=self.device)
model.load_state_dict(state)
model.eval()
self._model = model.to(self.device)
```

### Method: `classify(patch: np.ndarray) → SpaceStatus`

```python
pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
pil_img = pil_img.resize((input_size[1], input_size[0]))
tensor = TF.to_tensor(pil_img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = self._model(tensor)
    pred = int(torch.argmax(logits, dim=1).item())

return SpaceStatus(pred)
```

**IMPORTANT:** `CNNClassifier` imports `torch` at call time.
On Windows without the CPU-only PyTorch build, this causes `WinError 1114`.
This is why `ui.py` MUST NOT create `CNNClassifier` unless `method="cnn"` is explicit.

---

## Class: `SpaceClassifier` (Façade)

### Constructor

```python
SpaceClassifier(
    method: Literal["pixel_count", "cnn"] = "pixel_count",
    **kwargs
)
```

Selection logic:
```python
if method == "pixel_count":
    self._backend = PixelCountClassifier(**kwargs)
elif method == "cnn":
    self._backend = CNNClassifier(**kwargs)
else:
    raise ValueError(f"Unknown classifier method: {method!r}")
```

**Note:** `"overlap"` is NOT a valid method here. Overlap classification is handled
entirely in `pipeline.py` via `_detection_overlap()`. When `config.yaml` says
`method: "overlap"`, `ui.py` passes `method="pixel_count"` to `SpaceClassifier`
(the pixel_count backend serves as fallback when YOLO fails).

### Methods

```python
def classify(self, patch: np.ndarray) -> SpaceStatus:
    return self._backend.classify(patch)

def classify_batch(self, patches: list[np.ndarray]) -> list[SpaceStatus]:
    return [self.classify(p) for p in patches]
```

---

## Imports

```python
from enum import IntEnum
from pathlib import Path
from typing import Literal, Optional
import cv2
import numpy as np
# torch / torchvision imported lazily inside CNNClassifier methods
```

---

## Critical: `ui.py` Classifier Selection Logic

This is the correct logic in `ui.py._start_pipeline()` — it MUST remain exactly this:

```python
method = cls_cfg.get("method", "overlap")
if method == "cnn":
    # Only use CNN when explicitly requested
    classifier = SpaceClassifier(
        method="cnn",
        checkpoint_path=cls_cfg.get("cnn_checkpoint", "")
    )
else:
    # "overlap" and "pixel_count" both use pixel_count as SpaceClassifier backend
    # For "overlap": this serves as fallback when YOLO fails
    classifier = SpaceClassifier(
        method="pixel_count",
        pixel_threshold=cls_cfg.get("pixel_threshold", 0.25),
        blur_kernel=cls_cfg.get("blur_kernel", 5),
        adaptive_block=cls_cfg.get("adaptive_block", 11),
        adaptive_c=cls_cfg.get("adaptive_c", 2),
    )
```

**The bug that this fixes:** Before this fix, any non-"cnn" method (including "overlap")
fell into the `else` branch which tried to create `CNNClassifier`, imported torch,
and crashed with `WinError 1114 c10.dll` because CUDA torch was installed.

---

## Image Filtering Pipeline (Summary)

```
BGR patch
    │
    ├─ cv2.cvtColor(BGR → GRAY)           [Image Filtering: colour conversion]
    │
    ├─ cv2.GaussianBlur(ksize×ksize)      [Image Filtering: noise suppression]
    │
    ├─ cv2.adaptiveThreshold(             [Image Filtering: binarisation]
    │      ADAPTIVE_THRESH_GAUSSIAN_C,
    │      THRESH_BINARY_INV,
    │      block_size, C)
    │   OR cv2.threshold(OTSU) if patch too tiny
    │
    └─ count_nonzero / total_pixels       [Image Classification: pixel ratio decision]
           ratio ≥ pixel_threshold → OCCUPIED
           ratio < pixel_threshold → EMPTY
```
