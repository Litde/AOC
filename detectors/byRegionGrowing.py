import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SATURATION_VALUE = 2 #boost_saturation
LEVEL_LOW = 5 #apply_levels
LEVEL_HIGH = 95 #apply_levels

FLOOD_FILL_TOLERANCE = (10, 10, 10)
GRID_STEP = 50

MIN_AREA = 300 #contour

ASPECT_RATIO_TOLERANCE = 0.45 #square
MIN_CIRCULARITY = 0.6
MAX_CIRCULARITY = 1.4

def boost_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)

    h, s, v = cv2.split(hsv)

    h, s, v = cv2.split(hsv)
    s = np.clip(s * SATURATION_VALUE, 0, 255)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_levels(img):
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, (LEVEL_LOW, LEVEL_HIGH))
        out[:, :, c] = np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8)

def preprocess(img):
    img = boost_saturation(img)
    img = apply_levels(img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    return img

def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
    n = len(approx)

    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h) if h > 0 else 0

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
    return None

def detect_shapes_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        res = classify_shape(cnt)
        if res is not None:
            label, x, y, w, h, n = res
            detections.append((x, y, w, h, label, n))
    return detections

def draw_all_detections(img, detections):
    color_map = {
        "triangle": (0, 0, 255), "square": (255, 0, 0),
        "rectangle": (255, 128, 0), "circle": (0, 255, 0),
    }
    out = img.copy()
    for (x, y, w, h, label, n) in detections:
        color = color_map.get(label, (0, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def save_crops(img, detections, image_path, output_dir="cropped"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = []
    for idx, (x, y, w, h, shape_label, approxed) in enumerate(detections, start=1):
        H, W = img.shape[:2]
        cx = x + w / 2 cy = y + h / 2 scale = 1.3 new_w = w * scale new_h = h * scale x1 = int(cx - new_w / 2) y1 = int(cy - new_h / 2)
        x2 = int(cx + new_w / 2) y2 = int(cy + new_h / 2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

        if x1 >= x2 or y1 >= y2: continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        save_path = os.path.join(output_dir, f"{base}_{shape_label}_rg_{idx}.jpg")
        cv2.imwrite(save_path, crop)
        saved_paths.append(save_path)
    return saved_paths

def run_detector(image_path, printImages=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    prep_img = preprocess(img)
    h, w = prep_img.shape[:2]
    visited_mask = np.zeros((h + 2, w + 2), np.uint8)
    all_detections =  []

    # Iterate over the image with a grid to find seed points
    for y in range(0, h, GRID_STEP):
        for x in range(0, w, GRID_STEP):
            # If this pixel has not been filled yet
            if visited_mask[y + 1, x + 1] == 0:
                current_mask = np.zeros((h + 2, w + 2), np.uint8)
                
                cv2.floodFill(prep_img, current_mask, (x, y), 255,
                              loDiff=FLOOD_FILL_TOLERANCE, upDiff=FLOOD_FILL_TOLERANCE,
                              flags=cv2.FLOODFILL_MASK_ONLY)
                
                region_mask = current_mask[1:-1, 1:-1]
                detections = detect_shapes_from_mask(region_mask)
                
                if detections:
                    all_detections.extend(detections)
                
                visited_mask = cv2.bitwise_or(visited_mask, current_mask)
    
    list_of_crop_paths = save_crops(img, all_detections, image_path)

    if printImages:
        all_detections_img = draw_all_detections(img, all_detections)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(all_detections_img, cv2.COLOR_BGR2RGB))
        plt.title("Detections (Region Growing-based)")
        plt.axis("off")
        plt.show()

    return list_of_crop_paths

if __name__ == '__main__':
    # --- Konfiguracja ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    image_path = os.path.join(PARENT_DIR, "JPEGImages", "0000129.jpg")

    # --- Wczytanie i przetwarzanie obrazu ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {image_path}")
    else:
        print("1. Przetwarzanie wstępne obrazu...")
        prep_img = preprocess(img)

        print("2. Uruchamianie algorytmu 'Region Growing' (zalewanie)...")
        h = prep_img.shape[:2]
        visited_mask = np.zeros((h + 2, w + 2), np.uint8)
        all_detections = []

        for y in range(0, h, GRID_STEP):
            for x in range(0, w, GRID_STEP):
                if visited_mask[y + 1, x + 1] == 0:
                    current_mask = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(prep_img, current_mask, (x, y), 255,
                                  loDiff=FLOOD_FILL_TOLERANCE, upDiff=FLOOD_FILL_TOLERANCE,
                                  flags=cv2.FLOODFILL_MASK_ONLY)
                    
                    region_mask = current_mask[1:-1, 1:-1]
                    detections = detect_shapes_from_mask(region_mask)
                    
                    if detections:
                        all_detections.extend(detections)
                    
                    visited_mask = cv2.bitwise_or(visited_mask, current_mask)

        print("3. Rysowanie wszystkich detekcji...")
        img_with_detections = draw_all_detections(img, all_detections)
        final_mask = visited_mask[1:-1, 1:-1]

        # --- Wyświetlanie wyników krok po kroku ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Wizualizacja kroków detekcji (byRegionGrowing)", fontsize=16)

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Oryginalny obraz")
        axes[0].axis('off')

        axes[1].imshow(final_mask, cmap='gray')
        axes[1].set_title("Maska po 'Region Growing'")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Finalne detekcje")
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()