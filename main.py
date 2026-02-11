import os
from ClassificationModel import ClassificationRandomForest
from ClusterSVM import ClusterSVM

import detectors.byHough as hough
import detectors.byApproxPolyDP as approxPolyDP
import detectors.bySegmentation as segmentation
import detectors.byRegionGrowing as regionGrowing
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import cv2


def train_svm():
    model = ClusterSVM(target_class='rectangle', img_size=(64, 64))
    model.load_data()

    print(model.X.shape, model.y.shape)

    print("Training model...")

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    model.train_model(param_grid)
    model.save_model('models/binary_rectangle_model.joblib')

def train_rf():
    model = ClassificationRandomForest(data_dir='input/rectangle', img_size=(64, 64), n_estimators=100, max_depth=None, random_state=42)
    model.load_data()
    print("Training Random Forest model...")
    metrics = model.train_model()
    print("Training metrics:", metrics)
    model.save_model('models/rf_rectangle_model.joblib')

def test_svm(image_path, shape):
    model_path = f"models/binary_{shape}_model.joblib"
    model = ClusterSVM(target_class=shape, img_size=(64, 64))
    model.load_model(model_path)

    return model.predict(image_path)

def test_rf(image_path, shape):
    model_path = f"models/rf_{shape}_model.joblib"
    model = ClassificationRandomForest(data_dir='cropped/', img_size=(64, 64))
    model.load_model(model_path)

    return model.predict(image_path)


def get_true_label(image_filename: str):
    with open("labels/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = None
    for img in data["images"]:
        if img["file_name"] == image_filename:
            image_id = img["id"]
            break

    if image_id is None:
        raise ValueError(f"Nie znaleziono obrazu: {image_filename}")

    category_id_to_name = {
        cat["id"]: cat["name"]
        for cat in data["categories"]
    }

    category_names = set()
    for ann in data["annotations"]:
        if ann["image_id"] == image_id:
            cat_id = ann["category_id"]
            if cat_id in category_id_to_name:
                category_names.add(category_id_to_name[cat_id])
    return sorted(category_names)


def run_for_one(image_path):
    all_crops = []
    to_print = []

    modules = [
        ("hough", hough),
        ("approx", approxPolyDP),
        ("segmentation", segmentation),
        ("region_growing", regionGrowing)
    ]

    for name, mod in modules:
        to_print.append(f"Running {name}...")
        try:
            crops = mod.run_detector(image_path)
            all_crops.extend(crops)
        except Exception as e:
            to_print.append(f"Error in {name}: {e}")

    to_print.append("\nDETECTED SIGNS: ")
    printed_labels = set()
    detected_shape = None

    for filename in all_crops:
        shapes = ["circle", "triangle", "rectangle"]
        for shape in shapes:
            label = test_svm(filename, shape)
            if label == shape:
                detected_shape = shape
                break
        
        if detected_shape == None:
            return to_print
        
        label = test_rf(filename, detected_shape)


        if label.lower() in ("x-1.2", "x-1.1") or label in printed_labels:
            continue

        to_print.append(f"{label} - {os.path.basename(filename)}")
        printed_labels.add(label)

    return to_print


def run_for_many(n: int, random_pick: bool = False):
    image_dir = "JPEGImages"
    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if random_pick:
        images = random.sample(images, min(n, len(images)))
    else:
        images = images[:n]

    modules = [
        ("hough", hough),
        ("approx", approxPolyDP),
        ("segmentation", segmentation),
        ("region_growing", regionGrowing)
    ]

    ignored_labels = {"x-1.1", "x-1.2"}

    confusion_matrix = defaultdict(lambda: defaultdict(int))

    texts = []

    for image_name in images:
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path) or cv2.imread(image_path) is None:
            texts.append(f"Nie można otworzyć obrazu, pomijanie: {image_name}")
            continue

        try:
            true_labels = {
                lbl.lower() for lbl in get_true_label(image_name)
                if lbl.lower() not in ignored_labels
            }
        except Exception as e:
            texts.append(f"Brak etykiet dla {image_name}: {e}")
            continue

        all_crops = []

        for name, mod in modules:
            try:
                crops = mod.run_detector(image_path, printImages=False)
                all_crops.extend(crops)
            except Exception as e:
                texts.append(f"{image_name} - błąd w {name}: {e}")

        predicted_labels = set()

        detected_shape = None
        pred = None

        for filename in all_crops:
            shapes = ["circle", "triangle", "rectangle"]
            for shape in shapes:
                label = test_svm(filename, shape)
                if label == shape:
                    detected_shape = shape
                    break
            
            if detected_shape == None:
                pred = test_svm(filename, detected_shape)

            if pred is None:
                continue

            pred = pred.lower()
            if pred in ignored_labels:
                continue

            predicted_labels.add(pred)

        #True Positives
        true_positives = true_labels.intersection(predicted_labels)
        for lbl in true_positives:
            confusion_matrix[lbl][lbl] += 1

        #False Negatives
        false_negatives = true_labels.difference(predicted_labels)
        for lbl in false_negatives:
            confusion_matrix[lbl]['<brak_predykcji>'] += 1

        #False Positives
        false_positives = predicted_labels.difference(true_labels)
        for lbl in false_positives:
            confusion_matrix['<fałszywy_pozytyw>'][lbl] += 1

        texts.append(
            f"{image_name}: true={sorted(true_labels)}, pred={sorted(predicted_labels)}"
        )

    return texts, confusion_matrix
    
def calculate_and_plot_summary_metrics(cm: defaultdict):
    tp = 0
    fp = 0
    fn = 0

    true_labels = sorted([k for k in cm.keys() if k != "<fałszywy_pozytyw>"])

    for true_lbl in true_labels:
        preds = cm.get(true_lbl, {})
        #True Positive
        tp += preds.get(true_lbl, 0)
        #False Negative
        fn += preds.get('<brak_predykcji>', 0)

    fp += sum(cm.get("<fałszywy_pozytyw>", {}).values())

    print("\n--- Sumaryczna Macierz Pomyłek ---")
    print(f"  - True Positives (TP): {tp} (Poprawnie wykryte znaki)")
    print(f"  - False Positives (FP): {fp} (Błędne detekcje - nadmiarowe lub źle sklasyfikowane)")
    print(f"  - False Negatives (FN): {fn} (Niewykryte znaki)")
    print("  - True Negatives (TN): Nie dotyczy w zadaniach detekcji obiektów.")

    metrics = {'TP (Poprawne)': tp, 'FP (Błędne)': fp, 'FN (Niewykryte)': fn}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=['green', 'red', 'orange'])
    plt.ylabel('Liczba detekcji')
    plt.title('Podsumowanie wyników detekcji (TP, FP, FN)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')
    plt.show()


if __name__ == "__main__":
    #-------------------------------------------------------------------
    image_name = "0000010.jpg"
    detected_signs = run_for_one(f"JPEGImages\{image_name}")
    for sign in detected_signs:
        print(sign)

    categories = get_true_label( image_name)
    print(f"Kategorie na obrazie {image_name}:")
    for c in categories:
        print("-", c)


    #-------------------------------------------------------------------

    # texts, cm = run_for_many(40, random_pick=True)

    # for t in texts:
    #     print(t)

    # print("\nMACIERZ POMYŁEK:")
    # for true_lbl, preds in cm.items():
    #     for pred_lbl, count in preds.items():
    #         print(f"{true_lbl} -> {pred_lbl}: {count}")


    # calculate_and_plot_summary_metrics(cm)
    #-------------------------------------------------------------------
