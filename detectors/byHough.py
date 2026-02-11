import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

# preprocess
SATURATION_VALUE = 2
LEVEL_LOW = 2
LEVEL_HIGH = 98
GAUSSIAN_BLUR = 5

#hough linie
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 60
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 30

MIN_AREA = 800

#Hough kolka
HOUGH_CIRCLES_DP = 1
HOUGH_CIRCLES_MIN_DIST = 100
HOUGH_CIRCLES_PARAM1 = 300 # Górny próg
HOUGH_CIRCLES_PARAM2 = 50  #Próg akumulatora
HOUGH_CIRCLES_MIN_RADIUS = 10
HOUGH_CIRCLES_MAX_RADIUS = 700

#square ratio
ASPECT_RATIO_TOLERANCE = 0.5

INTERSECTION_CLUSTER_EPS = 25

CANNY_LOW_TRESHOLD = 75
CANNY_HIGH_TRESHOLD = 250


def boost_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * SATURATION_VALUE, 0, 255)
    hsv_enhanced = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_levels(img):
    output = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        min_val, max_val = np.percentile(ch, (LEVEL_LOW, LEVEL_HIGH))
        if max_val - min_val < 1e-3:
            output[:, :, c] = ch
        else:
            output[:, :, c] = np.clip((ch - min_val) * 255.0 / (max_val - min_val), 0, 255)
    return output.astype(np.uint8)


def preprocess(img):
    img = boost_saturation(img)
    img = apply_levels(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)
    edges = cv2.Canny(blurred, CANNY_LOW_TRESHOLD, CANNY_HIGH_TRESHOLD)
    return edges, blurred


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))


def cluster_points(points, eps=INTERSECTION_CLUSTER_EPS):
    clusters = []
    for p in points:
        placed = False
        for cl in clusters:
            cx, cy, count = cl
            if (p[0] - cx)**2 + (p[1] - cy)**2 <= eps**2:
                new_cx = (cx * count + p[0]) / (count + 1)
                new_cy = (cy * count + p[1]) / (count + 1)
                cl[0] = new_cx
                cl[1] = new_cy
                cl[2] = count + 1
                placed = True
                break
        if not placed:
            clusters.append([p[0], p[1], 1])
    centers = [(int(c[0]), int(c[1])) for c in clusters]
    return centers


def cluster_circles(circles, eps):
    clusters = []  #[cx, cy, r_sum, count]
    for (x, y, r) in circles:
        placed = False
        for cl in clusters:
            cx, cy, _, count = cl
            # Check distance to center
            if (x - cx) ** 2 + (y - cy) ** 2 <= eps ** 2:
                cl[0] = (cl[0] * count + x) / (count + 1)
                cl[1] = (cl[1] * count + y) / (count + 1)
                cl[2] += r
                cl[3] += 1
                placed = True
                break
        if not placed:
            clusters.append([float(x), float(y), float(r), 1])

    final_circles = [(int(c[0]), int(c[1]), int(c[2] / c[3])) for c in clusters]
    return final_circles


def cluster_polygons(detections, eps):
    clusters = []  #[cx, cy, w_sum, h_sum, count, label, n_vertices]
    for (x, y, w, h, label, n) in detections:
        px, py = x + w / 2, y + h / 2
        placed = False
        for cl in clusters:
            # only merge same-shaped polygons
            if cl[5] != label:
                continue

            cx, cy, _, _, count, _, _ = cl
            if (px - cx)**2 + (py - cy)**2 <= eps**2:
                cl[0] = (cl[0] * count + px) / (count + 1)
                cl[1] = (cl[1] * count + py) / (count + 1)
                cl[2] += w
                cl[3] += h
                cl[4] += 1
                placed = True
                break
        if not placed:
            clusters.append([px, py, float(w), float(h), 1, label, n])

    final_detections = []
    for cl in clusters:
        cx, cy, w_sum, h_sum, count, label, n = cl
        avg_w = int(w_sum / count)
        avg_h = int(h_sum / count)
        avg_x = int(cx - avg_w / 2)
        avg_y = int(cy - avg_h / 2)
        final_detections.append((avg_x, avg_y, avg_w, avg_h, label, n))
    return final_detections


def polygon_from_points(points):
    pts = np.array(points, dtype=np.int32)
    if pts.shape[0] < 3:
        return None
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def bbox_from_polygon(poly):
    x, y, w, h = cv2.boundingRect(poly)
    return x, y, w, h


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


def detect_polygons_from_lines(img, edges):
    output = img.copy()
    all_detections = []

    overlap_percent = 0.1
    h, w = edges.shape
    mid_x, mid_y = w // 2, h // 2
    overlap_x = int(mid_x * overlap_percent)
    overlap_y = int(mid_y * overlap_percent)
    tiles_coords = [
        (0, 0, mid_x + overlap_x, mid_y + overlap_y),  #Top-left
        (mid_x - overlap_x, 0, w, mid_y + overlap_y),  #Top-right
        (0, mid_y - overlap_y, mid_x + overlap_x, h),  #bot-left
        (mid_x - overlap_x, mid_y - overlap_y, w, h)  #bot-right
    ]

    for x1, y1, x2, y2 in tiles_coords:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        tile_edges = edges[y1:y2, x1:x2]
        tile_h, tile_w = tile_edges.shape

        if tile_edges.size < MIN_AREA:
            continue

        lines = cv2.HoughLinesP(tile_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)

        if lines is None or len(lines) < 3:
            continue

        lines = [l[0] for l in lines]

        MAX_LINES_FOR_COMBINATIONS = 100 
        if len(lines) < MAX_LINES_FOR_COMBINATIONS:
            MAX_SIGN_DIM = max(tile_h, tile_w) * 0.75
            MIN_TRIANGLE_SOLIDITY = 0.4

            for l1, l2, l3 in combinations(lines, 3):
                p12 = line_intersection(l1, l2)
                p23 = line_intersection(l2, l3)
                p31 = line_intersection(l3, l1)

                if p12 and p23 and p31:
                    poly = np.array([p12, p23, p31], dtype=np.int32)
                    area = cv2.contourArea(poly)
                    if area < MIN_AREA:
                        continue

                    x, y, w_poly, h_poly = cv2.boundingRect(poly)

                    if max(w_poly, h_poly) > MAX_SIGN_DIM or w_poly == 0 or h_poly == 0:
                        continue

                    solidity = area / (w_poly * h_poly)
                    if solidity < MIN_TRIANGLE_SOLIDITY:
                        continue

                    all_detections.append((x + x1, y + y1, w_poly, h_poly, "triangle", 3))

        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p = line_intersection(lines[i], lines[j])
                if p is not None and -tile_w <= p[0] <= 2 * tile_w and -tile_h <= p[1] <= 2 * tile_h:
                    intersections.append(p)

        if len(intersections) > 3:
            clustered = cluster_points(intersections)
            if len(clustered) == 4:  #4 vertex
                poly = polygon_from_points(clustered)
                if poly is not None and len(poly) == 4:
                    area = cv2.contourArea(poly)
                    if area < MIN_AREA:
                        continue

                    x, y, w_poly, h_poly = bbox_from_polygon(poly)
                    if h_poly > 0:
                        aspect_ratio = float(w_poly) / h_poly
                        if abs(1 - aspect_ratio) <= ASPECT_RATIO_TOLERANCE:
                            all_detections.append((x + x1, y + y1, w_poly, h_poly, "square", 4))

    if not all_detections:
        return output, []

    detections = cluster_polygons(all_detections, eps=HOUGH_CIRCLES_MIN_DIST / 2)

    for (x, y, w, h, label, n_vertices) in detections:
        color = (0, 0, 255) if label == "triangle" else (0, 255, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output, detections


def detect_circles_hough(img, blurred):
    output = img.copy()
    detections = []
    all_circles = []
    h, w = blurred.shape
    overlap_percent = 0.1

    mid_x, mid_y = w // 2, h // 2
    overlap_x = int(mid_x * overlap_percent)
    overlap_y = int(mid_y * overlap_percent)

    tiles_coords = [
        (0, 0, mid_x + overlap_x, mid_y + overlap_y),  # top-left
        (mid_x - overlap_x, 0, w, mid_y + overlap_y),  # Top-right
        (0, mid_y - overlap_y, mid_x + overlap_x, h),  # bot-left
        (mid_x - overlap_x, mid_y - overlap_y, w, h)  # bot-right
    ]

    for x1, y1, x2, y2 in tiles_coords:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        tile_blurred = blurred[y1:y2, x1:x2]

        if tile_blurred.size == 0:
            continue

        circles_in_tile = cv2.HoughCircles(tile_blurred, cv2.HOUGH_GRADIENT, dp=HOUGH_CIRCLES_DP,
                                           minDist=HOUGH_CIRCLES_MIN_DIST,
                                           param1=HOUGH_CIRCLES_PARAM1,
                                           param2=HOUGH_CIRCLES_PARAM2,
                                           minRadius=HOUGH_CIRCLES_MIN_RADIUS,
                                           maxRadius=HOUGH_CIRCLES_MAX_RADIUS)

        if circles_in_tile is not None:
            circles_in_tile = np.round(circles_in_tile[0, :]).astype("int")
            for (x_center, y_center, r) in circles_in_tile:
                all_circles.append((x_center + x1, y_center + y1, r))

    if not all_circles:
        return output, detections

    clustered_circles = cluster_circles(all_circles, eps=HOUGH_CIRCLES_MIN_DIST)

    for (x_center, y_center, r) in clustered_circles:
        x = x_center - r
        y = y_center - r
        w = h = 2 * r

        if w * h < MIN_AREA:
            continue

        detections.append((x, y, w, h, "circle", 0))
        cv2.circle(output, (x_center, y_center), r, (255, 0, 0), 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(output, "circle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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

        if shape_label == "circle":
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(w / 2)
            cv2.circle(output, center, radius, color, 2)
        else:
            #bounding box for polygons
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        # bounding box for polygons
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # label under bbox
        cv2.putText(
            output,
            f"{shape_label}",
            (x, y + h + 18),
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

    edges, blurred = preprocess(img)

    out_poly, poly_det = detect_polygons_from_lines(img, edges)

    out_circ, circ_det = detect_circles_hough(img, blurred)

    all_detections = poly_det + circ_det

    list_of_crop_paths = save_crops(img, all_detections, image_path)

    if printImages:
        #-------------------------------Printing----------------------------
        combined_img = draw_all_detections(img, all_detections)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours detection (Hough-based)")
        plt.axis("off")
        plt.show()
        #-------------------------------------------------------------------

    return list_of_crop_paths

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    image_path = os.path.join(PARENT_DIR, "JPEGImages", "0005911.jpg")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {image_path}")
    else:
        print("1. Przetwarzanie wstępne obrazu (canny, blur)...")
        edges, blurred = preprocess(img)

        print("1.5. Wykrywanie linii Hougha (do wizualizacji)...")
        lines = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
        img_with_lines = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print("2. Wykrywanie wielokątów (trójkąty, kwadraty)...")
        _, poly_det = detect_polygons_from_lines(img, edges)

        print("3. Wykrywanie okręgów...")
        _, circ_det = detect_circles_hough(img, blurred)

        print("4. Rysowanie wszystkich detekcji...")
        all_detections = poly_det + circ_det
        img_with_detections = draw_all_detections(img, all_detections)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle("Wizualizacja kroków detekcji (byHough)", fontsize=16)

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Oryginalny obraz")
        axes[0].axis('off')

        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title("Krawędzie Canny")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Wykryte linie Hougha")
        axes[2].axis('off')

        axes[3].imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Finalne detekcje")
        axes[3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
