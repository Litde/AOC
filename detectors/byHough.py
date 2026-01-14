import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# preprocess
SATURATION_VALUE = 2
LEVEL_LOW = 2
LEVEL_HIGH = 98
GAUSSIAN_BLUR = 5

# hough / geometry
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 45
HOUGH_MAX_LINE_GAP = 5

# contour filtering
MIN_AREA = 800

# circle Hough
HOUGH_CIRCLES_DP = 1
HOUGH_CIRCLES_MIN_DIST = 100
HOUGH_CIRCLES_PARAM1 = 400
HOUGH_CIRCLES_PARAM2 = 60
HOUGH_CIRCLES_MIN_RADIUS = 10
HOUGH_CIRCLES_MAX_RADIUS = 700

# square detection
ASPECT_RATIO_TOLERANCE = 0.5

# clustering distance for intersection points
INTERSECTION_CLUSTER_EPS = 8

CANNY_LOW_TRESHOLD = 50
CANNY_HIGH_TRESHOLD = 200


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
    """Compute intersection point of two lines given as (x1,y1,x2,y2).
    Returns (x,y) or None if parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))


def cluster_points(points, eps=INTERSECTION_CLUSTER_EPS):
    """Simple agglomerative clustering: group points within eps distance.
    Returns list of cluster centers (average points).
    """
    clusters = []
    for p in points:
        placed = False
        for cl in clusters:
            # compute distance to cluster center
            cx, cy, count = cl
            if (p[0] - cx)**2 + (p[1] - cy)**2 <= eps**2:
                # update centroid
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


def polygon_from_points(points):
    """Compute convex hull (polygon) from list of points and return as integer array."""
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


def detect_lines(edges):
    """Return list of lines from probabilistic Hough (x1,y1,x2,y2)."""
    raw = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                          minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
    if raw is None:
        return []
    lines = [tuple(l[0]) for l in raw]
    return lines


def detect_polygons_from_lines(img, edges):
    """
    Strategy:
    - use HoughLinesP to get many line segments
    - compute intersections of line pairs
    - cluster intersections to get stable vertices
    - compute convex hull of vertex set -> candidate polygon
    - if hull has 3 -> triangle, 4 -> check square aspect ratio -> square
    """
    output = img.copy()
    detections = []

    lines = detect_lines(edges)

    if len(lines) < 2:
        return output, detections

    # compute intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            p = line_intersection(lines[i], lines[j])
            if p is not None:
                # optionally discard intersections far outside image
                h, w = img.shape[:2]
                if -w <= p[0] <= 2*w and -h <= p[1] <= 2*h:
                    intersections.append(p)

    if not intersections:
        return output, detections

    #cluster intersections to reduce duplicates
    clustered = cluster_points(intersections)

    #compute convex hull of clustered points
    poly = polygon_from_points(clustered)
    if poly is None:
        return output, detections

    #filter small polygons by area
    area = cv2.contourArea(poly)
    if area < MIN_AREA:
        return output, detections

    #decide by number of hull vertices
    n_vertices = len(poly)
    x, y, w, h = bbox_from_polygon(poly)

    if n_vertices == 3:
        label = "triangle"
    elif n_vertices == 4:
        # check aspect ratio to be square-like
        aspect_ratio = float(w) / h if h != 0 else 0
        if abs(1 - aspect_ratio) <= ASPECT_RATIO_TOLERANCE:
            label = "square"
        else:
            return output, detections
    else:
        return output, detections

    detections.append((x, y, w, h, label, n_vertices))

    # draw polygon and bbox for visualization
    cv2.polylines(output, [poly], True, (0, 255, 255) if label == "square" else (0, 0, 255), 2)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255) if label == "square" else (0, 0, 255), 2)
    cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if label == "square" else (0, 0, 255), 2)

    return output, detections


def detect_circles_hough(img, blurred):
    """Detect circles via HoughCircles and return detection"""
    output = img.copy()
    detections = []

    # HoughCircles expects gray/blurred image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=HOUGH_CIRCLES_DP,
                               minDist=HOUGH_CIRCLES_MIN_DIST,
                               param1=HOUGH_CIRCLES_PARAM1,
                               param2=HOUGH_CIRCLES_PARAM2,
                               minRadius=HOUGH_CIRCLES_MIN_RADIUS,
                               maxRadius=HOUGH_CIRCLES_MAX_RADIUS)
    if circles is None:
        return output, detections

    circles = np.round(circles[0, :]).astype("int")
    for (x_center, y_center, r) in circles:
        x = x_center - r
        y = y_center - r
        w = h = 2 * r

        #area filter
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

        # bounding box
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

    #polygons via line intersections (triangles/squares)
    out_poly, poly_det = detect_polygons_from_lines(img, edges)

    #circles via HoughCircles
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
