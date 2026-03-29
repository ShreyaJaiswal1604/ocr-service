"""
Image enhancement for handwritten text before OCR.

For canvas renders (clean black on white):
  - Thicken strokes via dilation
  - Tight crop + padding
  - Resize to optimal resolution

For photos (noisy, uneven lighting):
  - Grayscale + denoise
  - Otsu binarization
  - Dilation
  - Noise cleanup
  - Crop + resize
"""

import cv2
import numpy as np
from PIL import Image


def enhance_for_ocr(image: Image.Image) -> Image.Image:
    """Auto-detect if canvas render or photo, apply appropriate pipeline."""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect if this is a clean canvas render (mostly white, sharp black strokes)
    # or a photo (varied colors, noise, uneven background)
    white_ratio = np.sum(gray > 240) / gray.size
    is_canvas = white_ratio > 0.7

    if is_canvas:
        result = _enhance_canvas(gray)
    else:
        result = _enhance_photo(gray)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))


def _enhance_canvas(gray: np.ndarray) -> np.ndarray:
    """
    Enhancement for clean canvas renders.
    Strokes are already black on white — just thicken and crop.
    """
    # Simple threshold — strokes are dark, background is white
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert: white strokes on black (for dilation)
    inverted = cv2.bitwise_not(binary)

    # Dilate — thicken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(inverted, kernel, iterations=2)

    # Back to black on white
    result = cv2.bitwise_not(dilated)

    # Crop and resize (use dilated mask so crop includes thickened strokes)
    result = _crop_and_resize(result, dilated)

    return result


def _enhance_photo(gray: np.ndarray) -> np.ndarray:
    """
    Enhancement for photos of handwritten text.
    Handles noise, uneven lighting, low contrast.
    """
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)

    # Otsu binarization (global, works well for photos with decent contrast)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert
    inverted = cv2.bitwise_not(binary)

    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    # Remove small noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 25:
            dilated[labels == i] = 0

    # Back to black on white
    result = cv2.bitwise_not(dilated)

    # Crop and resize
    result = _crop_and_resize(result, dilated)

    return result


def _crop_and_resize(
    result: np.ndarray,
    mask: np.ndarray,
    padding: int = 40,
    max_edge: int = 1540,
) -> np.ndarray:
    """Tight crop around content + resize to optimal OCR resolution."""
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(result.shape[1], x + w + padding)
        y2 = min(result.shape[0], y + h + padding)
        result = result[y1:y2, x1:x2]

    h, w = result.shape[:2]
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif max(h, w) < max_edge // 2:
        scale = (max_edge // 2) / max(h, w)
        result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    return result
