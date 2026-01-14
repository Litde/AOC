import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SATURATION_VALUE = 2
LEVEL_LOW = 5
LEVEL_HIGH = 95
GAUSSIAN_BLUR = 5

# contour filtering
MIN_AREA = 400

# classification
ASPECT_RATIO_TOLERANCE = 0.45 #square
MIN_CIRCULARITY = 0.8
MAX_CIRCULARITY = 1.1

# saliency
SALIENCY_BLUR = 5
SALIENCY_THRESH = 65


def boost_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * SATURATION_VALUE, 0, 255)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_levels(img):
    """Histogram stretching based on percentiles."""
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, (LEVEL_LOW, LEVEL_HIGH))
        out[:, :, c] = np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8)

def preprocess(img):
    img = boost_saturation(img)
    img = apply_levels(img)

    # improved HLS color normalization
    hls_norm = improved_hls_normalization(img)

    #merge normalized L with original color for saliency stability
    luminance = (hls_norm[:, :, 1] * 255).astype(np.uint8)
    img = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)

    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

    return img

def improved_hls_normalization(img):
    """Enhanced HLS normalization inspired by Hanbury & Serra."""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    H, L, S = cv2.split(hls)

    # normalize Hue to 0–1
    H = H / 180.0

    # normalize Saturation adaptively
    S = S / (np.max(S) + 1e-6)

    #gamma compression for luminance
    L = L / 255.0
    L = np.sqrt(L)     # lighten dark, darker bright

    out = cv2.merge([H, L, S])
    return out


def get_saliency_mask(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, sal_map = saliency.computeSaliency(img)

    sal_map = (sal_map * 255).astype(np.uint8)

    # smooth and threshold
    sal_blurred = cv2.GaussianBlur(sal_map, (SALIENCY_BLUR, SALIENCY_BLUR), 0)
    _, mask = cv2.threshold(sal_blurred, SALIENCY_THRESH, 255, cv2.THRESH_BINARY)

    # cleanup ------------------------------------------
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    #----------------------------------------------------

    return mask, sal_map

def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    n = len(approx)

    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)

    # circle
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
    if n > 5 and MIN_CIRCULARITY < circularity < MAX_CIRCULARITY:
        return "circle", x, y, w, h, n

    # triangle
    if n == 3:
        return "triangle", x, y, w, h, n

    # quadrilaterals
    if n == 4:
        if abs(1 - aspect_ratio) < ASPECT_RATIO_TOLERANCE:
            return "square", x, y, w, h, n
        else:
            return "rectangle", x, y, w, h, n


    if n >= 8 and n < 10:
        return "octagon", x, y, w, h, n

    return None

def detect_shapes(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        res = classify_shape(cnt)
        if res is None:
            continue
        label, x, y, w, h, n = res
        detections.append((x, y, w, h, label, n))

    return detections


def draw_all_detections(img, detections):
    color_map = {
        "triangle": (0, 0, 255),
        "square": (255, 0, 0),
        "rectangle": (255, 128, 0),
        "circle": (0, 255, 0),
        "octagon": (128, 0, 128),

    }
    out = img.copy()
    for (x, y, w, h, label, n) in detections:
        color = color_map.get(label, (0, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def save_crops(img, detections, image_path, output_dir="cropped"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    paths = []

    for idx, (x, y, w, h, label, n) in enumerate(detections, start=1):
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        p = os.path.join(output_dir, f"{base}_{label}_{idx}.jpg")
        cv2.imwrite(p, crop)
        paths.append(p)
    return paths

def run_detector(image_path, printImages=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    prep = preprocess(img)
    mask, sal_map = get_saliency_mask(prep)

    detections = detect_shapes(prep, mask)
    list_of_crop_paths = save_crops(img, detections, image_path)

    if printImages:
        all_detections = draw_all_detections(img, detections)

        # plt.figure(figsize=(24, 12))
        # plt.subplot(131)
        # plt.title("Saliency Map")
        # plt.imshow(sal_map, cmap="gray")
        # plt.axis("off")
        #
        # plt.subplot(132)
        # plt.title("Saliency Mask")
        # plt.imshow(mask, cmap="gray")
        # plt.axis("off")
        #
        # plt.subplot(133)
        # plt.title("Detected Shapes")
        # plt.imshow(cv2.cvtColor(all_detections, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        #
        # plt.show()

        #-------------------------------Printing----------------------------
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(all_detections, cv2.COLOR_BGR2RGB))
        plt.title("Contours detection (Segmentation-based)")
        plt.axis("off")
        plt.show()
        #-------------------------------------------------------------------

    return list_of_crop_paths
