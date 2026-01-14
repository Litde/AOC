import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#preprocess
SATURATION_VALUE = 2
LEVEL_LOW = 2
LEVEL_HIGH = 98
GAUSSIAN_FILTER_SIZE = 3
GAUSSIAN_FILTER_THRESHOLD = 0

#contour filtering
MIN_AREA = 800
COLOR_VARIANCE_THRESHOLD = 0

#square detection
ASPECT_RATIO_TOLERANCE = 0.2

#circle detection
MIN_CIRCULARITY = 0.5
MAX_CIRCULARITY = 1.2

CANNY_LOW_TRESHOLD = 50
CANNY_HIGH_TRESHOLD = 300

def boost_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * SATURATION_VALUE, 0, 255)
    hsv_enhanced = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_levels(img):
    """Histogram stretching based on percentiles."""
    output = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        min_val, max_val = np.percentile(ch, (LEVEL_LOW, LEVEL_HIGH))
        output[:, :, c] = np.clip((ch - min_val) * 255.0 / (max_val - min_val), 0, 255)
    return output.astype(np.uint8)

def preprocess(img):
    img = boost_saturation(img)
    img = apply_levels(img)
    filtered = cv2.bilateralFilter(img, 9, 60, 30)
    edges = cv2.Canny(filtered, CANNY_LOW_TRESHOLD, CANNY_HIGH_TRESHOLD)
    return edges

def save_crops(img, detections, image_path, output_dir="cropped"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    saved_paths = []
    for idx, (x, y, w, h, shape_label, approxed) in enumerate(detections, start=1):

        H, W = img.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        if x1 >= x2 or y1 >= y2:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        save_path = os.path.join(output_dir, f"{base}_{shape_label}_{idx}.jpg")
        cv2.imwrite(save_path, crop)
        saved_paths.append(save_path)

    return saved_paths


def detect_triangles(img, edges):
    output = img.copy()
    detections = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(approx)
            detections.append((x, y, w, h, "triangle", len(approx)))

            color = (0, 0, 255)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output, "triangle", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    return output, detections

def detect_squares(img, edges):
    output = img.copy()
    detections = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if abs(1 - aspect_ratio) > ASPECT_RATIO_TOLERANCE:
                continue

            detections.append((x, y, w, h, "square", len(approx)))

            color = (0, 255, 255)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output, "square", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    return output, detections


def detect_circles(img, edges):
    output = img.copy()
    detections = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if len(approx) > 5:
            circularity = 4 * np.pi * cv2.contourArea(cnt) / (perimeter ** 2)
            if MIN_CIRCULARITY < circularity <= MAX_CIRCULARITY:

                x, y, w, h = cv2.boundingRect(approx)
                detections.append((x, y, w, h, "circle", len(approx)))

                color = (255, 0, 0)
                cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                cv2.putText(output, "circle", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return output, detections


def draw_all_detections(img, detections):
    output = img.copy()
    shape_colors = {
        "triangle": (0, 0, 255),
        "square":   (255, 0, 0),
        "circle":   (0, 255, 0)
    }
    for (x, y, w, h, shape_label, n_points) in detections:
        color = shape_colors.get(shape_label, (0, 255, 0))

        #bounding box
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            output,
            shape_label,
            (x, y + h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output

def run_detector(image_path, printImages=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    edges = preprocess(img)

    _, tri_det = detect_triangles(img, edges)
    _, sq_det = detect_squares(img, edges)
    _, circ_det = detect_circles(img, edges)

    all_detections = tri_det + sq_det + circ_det

    list_of_crop_paths = save_crops(img, all_detections, image_path)

    if printImages:
        #-------------------------------Printing----------------------------
        combined_img = draw_all_detections(img, all_detections)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Contours detectetion by number of approxPolyDP")
        plt.axis("off")
        plt.show()
        #-------------------------------------------------------------------

    return list_of_crop_paths
