from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = Path("/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/seals")   # change this
OUT_CANNY = Path("/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/seals_proc_canny")
OUT_BINARY = Path("/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/seals_proc_binary")
OUT_GRAD = Path("/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/seals_proc_grad")
for out_dir in [OUT_CANNY, OUT_BINARY, OUT_GRAD]:
    out_dir.mkdir(parents=True, exist_ok=True)


def preprocess_base(gray: np.ndarray) -> np.ndarray:
    # Preserve larger structure, suppress small corrosion texture
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(smooth)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    return blur


valid_exts = (".jpg", ".png", ".jpeg")

count = 0
for img_path in tqdm(sorted(INPUT_DIR.glob("*")), desc="Processing images"):
    if img_path.suffix.lower() not in valid_exts:
        print(f"Skipping non-image file: {img_path}")
        continue
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping unreadable file: {img_path}")
        continue

    base = preprocess_base(img)

    # 1) Smoothed Canny
    canny = cv2.Canny(base, 40, 120)
    canny = cv2.dilate(canny, np.ones((2, 2), np.uint8), iterations=1)
    canny = 255 - canny
    cv2.imwrite(str(OUT_CANNY / img_path.name), canny)

    # 2) Adaptive/binary structural map
    binary = cv2.adaptiveThreshold(
        base,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    binary = 255 - binary
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cv2.imwrite(str(OUT_BINARY / img_path.name), binary)

    # 3) Morphological gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.morphologyEx(base, cv2.MORPH_GRADIENT, kernel)
    _, grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad = 255 - grad
    cv2.imwrite(str(OUT_GRAD / img_path.name), grad)
    count += 1

print(f"Total images processed: {count}")