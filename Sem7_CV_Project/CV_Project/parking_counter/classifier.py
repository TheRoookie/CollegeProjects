"""
classifier.py
-------------
Binary classifier that determines whether a segmented parking-space
patch is **Empty (0)** or **Occupied (1)**.

Two interchangeable back-ends are provided:

* ``"pixel_count"`` (default) – classic OpenCV pipeline:
    1. Convert the cropped ROI patch to grayscale.        [Image Filtering]
    2. Apply Gaussian blur to suppress noise.             [Image Filtering]
    3. Apply Adaptive Thresholding to binarise.           [Image Filtering]
    4. Count non-zero pixels; if the fraction exceeds a
       configurable threshold, the space is "occupied".   [Image Classification]

* ``"cnn"`` – a minimal PyTorch CNN loaded from a saved checkpoint.
    Feed the patch through a small convolutional network fine-tuned
    on top-down parking images.                           [Image Classification]

Syllabus topics covered: **Image Filtering**, **Image Classification**
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np


class SpaceStatus(IntEnum):
    EMPTY = 0
    OCCUPIED = 1


# ============================================================
# Pixel-count classifier (no GPU required, highly interpretable)
# ============================================================

class PixelCountClassifier:
    """
    Classifies a parking patch using adaptive thresholding + pixel counting
    with the **correct inverted decision rule** for textured surfaces.

    Algorithm (Image Filtering → Image Classification)
    ---------------------------------------------------
    1. Grayscale conversion.
    2. Small Gaussian blur (kernel=3) — suppresses single-pixel video
       compression noise without erasing brick-mortar detail.
    3. Adaptive Thresholding — ``ADAPTIVE_THRESH_GAUSSIAN_C``,
       ``THRESH_BINARY_INV``, ``blockSize=25``, ``C=16``.

       Mathematical basis: each pixel p is compared to the weighted
       Gaussian average of its (blockSize × blockSize) neighbourhood μₚ:

           dst(p) = 255  if src(p) < μₚ − C  else  0

       A pixel is white (255) only when it is *locally dark* — i.e.
       darker than its immediate surroundings by more than C grey-levels.

    4. Morphological dilation (5×5 ellipse, 1 iteration) — connects
       nearby white pixels so brick-mortar lines merge into filled blobs
       rather than scattered dots.

    5. Non-zero pixel count → occupancy decision.

    Why the rule is **low count = OCCUPIED**
    -----------------------------------------
    After THRESH_BINARY_INV on cobblestone/brick:
    • Each brick tile is *darker* than its lighter mortar border → that
      tile's pixels survive the threshold → **HIGH white-pixel count**.
    • A car body is a large, locally-uniform region (smooth paint, few
      internal intensity transitions) → most pixels are close to μₚ →
      very few survive the threshold → **LOW white-pixel count**.

    Therefore:
        count < threshold  →  smooth surface  →  OCCUPIED
        count ≥ threshold  →  textured surface →  EMPTY

    Parameters
    ----------
    count_threshold : int
        Absolute number of white pixels below which a patch is OCCUPIED.
        Should be tuned to the slot pixel size.  Default 900 is calibrated
        for patches of roughly 60×100 pixels at 640 px processing width.
    blur_kernel : int
        Gaussian blur kernel size (must be odd).  Keep small (3–5).
    adaptive_block : int
        Block size for adaptive threshold (must be odd, ≥ 3).  Default 25
        works well for bricks ≈ 15–30 px wide.
    adaptive_c : int
        Constant subtracted from local mean.  Default 16 — conservative
        enough to ignore compression noise yet sensitive to mortar lines.
    """

    def __init__(
        self,
        count_threshold: int = 900,
        blur_kernel: int = 3,
        adaptive_block: int = 25,
        adaptive_c: int = 16,
        # kept for backward-compat with config keys
        pixel_threshold: float = 0.0,
        texture_threshold: float = 0.0,
        blob_area_fraction: float = 0.0,
    ) -> None:
        self.count_threshold = count_threshold
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.adaptive_block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
        self.adaptive_c = adaptive_c

    def classify(self, patch: np.ndarray) -> SpaceStatus:
        """
        Classify *patch* as empty or occupied.

        Pipeline: grayscale → small blur → adaptive threshold (BINARY_INV,
        block=25, C=16) → dilation → count non-zero pixels.

        Decision rule (inverted from naive pixel-counting):
            count < count_threshold  →  OCCUPIED
            count ≥ count_threshold  →  EMPTY

        Rationale: cobblestone/brick produces many locally-dark pixels
        (HIGH count).  A car body is locally uniform (LOW count).

        Parameters
        ----------
        patch : np.ndarray
            Cropped BGR image of a single parking spot.

        Returns
        -------
        SpaceStatus
            ``SpaceStatus.EMPTY`` or ``SpaceStatus.OCCUPIED``.
        """
        # Guard: reject None or degenerate patches
        if patch is None or patch.ndim < 2:
            return SpaceStatus.EMPTY
        ph, pw = patch.shape[:2]
        if ph < 5 or pw < 5:
            return SpaceStatus.EMPTY

        # Step 1 — Grayscale (Image Filtering)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch

        # Step 2 — Small Gaussian blur: removes compression artefacts only,
        # preserves brick-mortar detail (Image Filtering)
        ksize = min(self.blur_kernel, ph, pw)
        ksize = ksize if ksize % 2 == 1 else max(1, ksize - 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Step 3 — Adaptive Threshold (Image Segmentation)
        # blockSize must be odd and ≤ patch dimension
        block = min(self.adaptive_block, ph if ph % 2 == 1 else ph - 1,
                                         pw if pw % 2 == 1 else pw - 1)
        block = max(3, block)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,   # locally-dark pixels become WHITE
            block,
            self.adaptive_c,
        )

        # Step 4 — Dilation: merge nearby white pixels so brick mortar
        # lines connect into filled regions (morphological operation)
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.dilate(thresh, dil_k, iterations=1)

        # Step 5 — Count and decide (Image Classification)
        count = int(np.count_nonzero(thresh))
        # Scale threshold by actual patch area relative to expected 60×100 base
        base_area = 6000  # pixels at 640 px processing width
        scaled_threshold = int(self.count_threshold * (ph * pw) / base_area)
        scaled_threshold = max(100, scaled_threshold)  # never go below 100

        return SpaceStatus.EMPTY if count >= scaled_threshold else SpaceStatus.OCCUPIED

    def preprocess_for_display(self, patch: np.ndarray) -> np.ndarray:
        """
        Return intermediate binary image for debug / viva visualisation.
        Shows adaptive-threshold + dilation result: many white pixels = empty,
        few white pixels = occupied.
        """
        if patch is None or patch.size == 0:
            return np.zeros((50, 50), dtype=np.uint8)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        ph, pw = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        block = min(self.adaptive_block, ph if ph % 2 == 1 else ph - 1,
                                         pw if pw % 2 == 1 else pw - 1)
        block = max(3, block)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, self.adaptive_c,
        )
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(thresh, dil_k, iterations=1)


# ============================================================
# CNN classifier (PyTorch, optional heavier back-end)
# ============================================================

class CNNClassifier:
    """
    Lightweight CNN classifier for parking spot occupancy.

    The network is a small custom ConvNet with two convolutional layers
    followed by a fully-connected head producing a binary output.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a ``*.pt`` file saved with ``torch.save(model.state_dict(), ...)``.
    input_size : tuple[int, int]
        (height, width) the network expects.  All patches are resized to this.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        input_size: tuple[int, int] = (64, 64),
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.input_size = input_size
        self.device = device
        self._model = None

    def _build_model(self):
        """Construct the tiny CNN architecture."""
        import torch
        import torch.nn as nn

        class ParkingCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return ParkingCNN()

    def load(self) -> None:
        """Load the CNN weights from checkpoint."""
        import torch
        model = self._build_model()
        state = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self._model = model.to(self.device)

    def classify(self, patch: np.ndarray) -> SpaceStatus:
        """Classify a BGR patch using the loaded CNN."""
        import torch
        import torchvision.transforms.functional as TF  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        if self._model is None:
            self.load()

        pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((self.input_size[1], self.input_size[0]))
        tensor = TF.to_tensor(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            pred = int(torch.argmax(logits, dim=1).item())

        return SpaceStatus(pred)


# ============================================================
# Unified SpaceClassifier façade
# ============================================================

class SpaceClassifier:
    """
    Façade that selects the correct back-end based on configuration.

    Parameters
    ----------
    method : "pixel_count" | "cnn"
        Which classifier to use.
    **kwargs
        Forwarded to the underlying classifier constructor.
    """

    def __init__(
        self,
        method: Literal["pixel_count", "cnn"] = "pixel_count",
        **kwargs,
    ) -> None:
        if method == "pixel_count":
            # Forward only keys PixelCountClassifier accepts; ignore extras
            _pc_keys = {
                "count_threshold", "blur_kernel", "adaptive_block", "adaptive_c",
                # backward-compat keys (accepted but unused in new pipeline)
                "pixel_threshold", "texture_threshold", "blob_area_fraction",
            }
            pc_kwargs = {k: v for k, v in kwargs.items() if k in _pc_keys}
            self._backend = PixelCountClassifier(**pc_kwargs)
        elif method == "cnn":
            self._backend = CNNClassifier(**kwargs)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown classifier method: {method!r}")

    def classify(self, patch: np.ndarray) -> SpaceStatus:
        """Classify a single parking-space patch."""
        return self._backend.classify(patch)

    def classify_batch(
        self, patches: list[np.ndarray]
    ) -> list[SpaceStatus]:
        """Classify a list of patches and return a list of statuses."""
        return [self.classify(p) for p in patches]
