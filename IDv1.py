import cv2
import numpy as np
import imutils

def preprocess(img, target_width=1000):
    img = imutils.resize(img, width=target_width)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    return blur

def hsv_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Zwiększamy progi nasycenia i jasności – tylko jaskrawe obiekty
    lower_red1 = np.array([0, 120, 120]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 120]); upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    lower_blue = np.array([90, 120, 80]); upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_y = np.array([15, 120, 120]); upper_y = np.array([35, 255, 255])
    mask_y = cv2.inRange(hsv, lower_y, upper_y)

    combined = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_blue), mask_y)
    return combined

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3,3), 0)
    return mask

def has_inner_contrast(img_crop):
    """Sprawdza, czy wnętrze potencjalnego znaku ma silny kontrast."""
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > 100  # wariancja Laplace’a


def contour_candidates(mask, img,
                       min_area:int=700, max_area_ratio:float=0.2,
                       circularity_thresh:float=0.0, solidity_thresh:float=0.0,
                       color_fraction_thresh:float=0.0,
                       allow_shapes:list=["circle","rectangle","triangle","polygon"]) -> list:
    """
    Wyszukuje kontury odpowiadające znakom drogowym (okrągłym, prostokątnym, trójkątnym, wielokątnym).
    """
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    h, w = img.shape[:2]
    filtered = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > (h * w * max_area_ratio):
            continue

        x, y, ww, hh = cv2.boundingRect(c)
        if ww == 0 or hh == 0:
            continue
        ar = ww / float(hh)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        rect_area_ratio = area / float(ww * hh) if ww*hh > 0 else 0
        if rect_area_ratio < 0.45:
            continue

        mask_crop = mask[y:y+hh, x:x+ww]
        color_fraction = cv2.countNonZero(mask_crop) / float(ww * hh) if ww*hh > 0 else 0

        # --- kształt ---
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        vertices = len(approx)
        shape_type = None

        if vertices == 3 and "triangle" in allow_shapes:
            shape_type = "triangle"

        elif vertices == 4 and "rectangle" in allow_shapes:
            # prostokąt lub kwadrat
            if 0.5 <= ar <= 1.5:
                shape_type = "rectangle"

        elif 5 <= vertices <= 8 and "polygon" in allow_shapes:
            shape_type = f"polygon_{vertices}"

        elif circularity > 0.7 and "circle" in allow_shapes:
            shape_type = "circle"

        # Jeśli żaden z typów nie pasuje
        if shape_type is None:
            continue

        # Progi wspólne
        if circularity < circularity_thresh or solidity < solidity_thresh or color_fraction < color_fraction_thresh:
            continue

        crop = img[y:y+hh, x:x+ww]
        if not has_inner_contrast(crop):
            continue

        filtered.append((c, x, y, ww, hh, shape_type))

    return filtered



def detect_signs(image_path:str, min_area:int=500,
                 circularity:float=0.65, solidity:float=0.85,
                 color_fraction:float=0.25, debug:bool=True) -> list:
    """
        Wykrywa znaki na obrazie wskazanym przez `image_path`.

        Parametry:
        - image_path (str): ścieżka do pliku obrazka.
        - min_area (float): minimalna powierzchnia konturu uwzględniana jako kandydat.
        - circularity (float): próg circularity (okrągłość) do filtrowania konturów.
        - solidity (float): próg solidności (wypełnienia względem otoczki).
        - color_fraction (float): minimalny udział maskowanego koloru wewnątrz bounding boxa.
        - debug (bool): jeśli True, pokazuje okna z maską oraz wykryciami (blokuje wykonanie do naciśnięcia klawisza).
        Zwraca:
        - lista wykryć w postaci krotek (contour, x, y, w, h). W przypadku uruchomienia jako skrypt
          funkcja wypisuje liczbę wykrytych znaków i wyświetla okna UI gdy `debug` jest True.
    """
    try:
        img = cv2.imread(image_path)
        img = preprocess(img)
        mask = hsv_masks(img)
        mask = clean_mask(mask)

        detections = contour_candidates(mask, img,
                                        min_area=min_area, max_area_ratio=0.2,
                                        circularity_thresh=circularity,
                                        solidity_thresh=solidity,
                                        color_fraction_thresh=color_fraction,
                                        allow_shapes=["circle", "triangle", "rectangle", "polygon"])

        output = img.copy()

        for (c, x, y, w, h, shape_type) in detections:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, shape_type, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if debug:
            cv2.imshow("Mask", mask)
            cv2.imshow("Detections", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Detected {len(detections)} signs.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        detections = []
    return detections

if __name__ == "__main__":
    detect_signs("JPEGImages/0000010.jpg", debug=True)


