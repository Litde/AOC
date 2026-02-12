import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SATURATION_VALUE = 2 #boost_saturation
LEVEL_LOW = 5 #apply_levels
LEVEL_HIGH = 95 #apply_levels

MIN_AREA = 500 #contour filtering
ASPECT_RATIO_TOLERANCE = 0.5 #square
MIN_CIRCULARITY = 0.7
MAX_CIRCULARITY = 1.2

# saliency
SALIENCY_BLUR = 3
SALIENCY_THRESH = 40


def boost_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
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

    # improved HLS color normalization
    hls_norm = improved_hls_normalization(img)

    #merge normalized L with original color for saliency stability
    luminance = (hls_norm[:, :, 1] * 255).astype(np.uint8)
    img = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)

    img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

    return img

def improved_hls_normalization(img):
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

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=4)
    

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


    # if n >= 8 and n < 10:
    #     return "octagon", x, y, w, h, n

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
    color = (255, 0, 255) # Pink
    out = img.copy()
    for (x, y, w, h, label, n) in detections:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 10)
        cv2.putText(out, label, (x, y + h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 10)
    return out

def save_crops(img, detections, image_path, output_dir="cropped"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    saved_paths = []
    for idx, (x, y, w, h, shape_label, approxed) in enumerate(detections, start=1):

        H, W = img.shape[:2]

        cx = x + w / 2
        cy = y + h / 2
        scale = 1.3
        new_w = w * scale
        new_h = h * scale
        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x2 = int(cx + new_w / 2)
        y2 = int(cy + new_h / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if x1 >= x2 or y1 >= y2:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        save_path = os.path.join(output_dir, f"{base}_{shape_label}_{idx}.jpg")
        cv2.imwrite(save_path, crop)
        saved_paths.append(save_path)

    return saved_paths

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

        # plt.subplot(133)
        # plt.title("Detected Shapes")
        # plt.imshow(cv2.cvtColor(all_detections, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        #
        # plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(all_detections, cv2.COLOR_BGR2RGB))
        plt.title("Contours detection (Segmentation-based)")
        plt.axis("off")
        plt.show()
        #-------------------------------------------------------------------

    return list_of_crop_paths

if __name__ == '__main__':
    # --- Konfiguracja ---
    # Użyj przykładowego obrazu do testowania.
    # Zakładamy, że skrypt jest w AOC/AOC/detectors, a obrazy w AOC/AOC/JPEGImages
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    image_path = os.path.join(PARENT_DIR, "JPEGImages", "0000339.jpg")

    # --- Wczytanie i przetwarzanie obrazu ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {image_path}")
    else:
        print("1. Przetwarzanie wstępne obrazu...")
        prep_img = preprocess(img)

        print("2. Generowanie mapy saliencji i maski binarnej...")
        # get_saliency_mask zwraca:
        # - mask: finalna maska binarna po operacjach morfologicznych (erozja, dylatacja)
        # - sal_map: surowa mapa saliencji (wynik segmentacji)
        mask, sal_map = get_saliency_mask(prep_img)

        print("3. Wykrywanie kształtów na podstawie maski...")
        detections = detect_shapes(prep_img, mask)

        print("4. Rysowanie wykrytych obiektów (bounding box)...")
        img_with_detections = draw_all_detections(img, detections)

        # --- Wizualizacja posegmentowanych obszarów ---
        # Znajdź kontury na masce, aby zwizualizować każdy obszar osobno.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Stwórz czarny obraz do narysowania kolorowych konturów.
        segmented_areas_vis = np.zeros_like(img)
        for cnt in contours:
            # Narysuj każdy znaleziony obszar (kontur) wypełniony losowym kolorem.
            color = tuple(np.random.randint(60, 256, 3).tolist())
            cv2.drawContours(segmented_areas_vis, [cnt], -1, color, -1)

        # --- Wyświetlanie wyników krok po kroku ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        fig.suptitle("Wizualizacja kroków detekcji (bySegmentation)", fontsize=16)

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Oryginalny obraz")
        axes[0].axis('off')

        axes[1].imshow(sal_map, cmap='gray')
        axes[1].set_title("Mapa istotności")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(segmented_areas_vis, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Posegmentowane obszary")
        axes[2].axis('off')

        axes[3].imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Finalne detekcje")
        axes[3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
