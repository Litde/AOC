import os
from ClassificationModel import ClassificationRandomForest
from ClusterSVM import ClusterSVM

import detectors.byHough as hough
import detectors.byApproxPolyDP as approxPolyDP
import detectors.bySegmentation as segmentation
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt


def train_svm(target_class='rectangle'):
    model = ClusterSVM(target_class=target_class, img_size=(64, 64))
    model.load_data()

    print(model.X.shape, model.y.shape)

    print("Training model...")

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    model.train_model(param_grid)
    model.save_model(f'models/binary_{target_class}_model.joblib')

def train_rf(target_class='rectangle'):
    model = ClassificationRandomForest(data_dir=f'enhanced_input/{target_class}', img_size=(64, 64), n_estimators=100, max_depth=None, random_state=42)
    model.load_data()
    print("Training Random Forest model...")
    metrics = model.train_model()
    print("Training metrics:", metrics)
    model.save_model(f'models/rf_{target_class}_model.joblib')

def test_svm():
    model = ClusterSVM(target_class='rectangle', img_size=(64, 64))
    model.load_model('models/binary_rectangle_model.joblib')

    y_pred = model.predict('image.png')
    print("Predicted label:", y_pred)

    texts = []

    for image_name in images:
        image_path = os.path.join(image_dir, image_name)

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

        for filename in all_crops:
            if "circle" in filename or "octagon" in filename:
                shape = "circle"
            elif "triangle" in filename:
                shape = "triangle"
            elif "square" in filename or "rectangle" in filename:
                shape = "rectangle"
            else:
                continue

            pred = test_svm(filename, shape)

            if pred is None:
                continue

            pred = pred.lower()
            if pred in ignored_labels:
                continue

            predicted_labels.add(pred)

        # Prawidłowo zidentyfikowane znaki (True Positives)
        true_positives = true_labels.intersection(predicted_labels)
        for lbl in true_positives:
            confusion_matrix[lbl][lbl] += 1

        # Pominięte znaki (False Negatives)
        false_negatives = true_labels.difference(predicted_labels)
        for lbl in false_negatives:
            confusion_matrix[lbl]['<brak_predykcji>'] += 1

        # Błędnie zidentyfikowane znaki (False Positives)
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

    # Obliczanie TP i FN na podstawie prawdziwych etykiet
    for true_lbl in true_labels:
        preds = cm.get(true_lbl, {})
        # Poprawnie wykryty i sklasyfikowany znak (True Positive)
        tp += preds.get(true_lbl, 0)
        # Niewykryty znak (False Negative)
        fn += preds.get('<brak_predykcji>', 0)

    # Obliczanie FP na podstawie detekcji, które nie miały odpowiednika w prawdziwych etykietach
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

    y_pred = model.predict('image.png')
    print("Predicted label:", y_pred)

if __name__ == "__main__":
    # train_rf()
    train_svm(target_class='rectangle')
    # test_svm()
    # train_rf()
    # test_rf()

    # detected_signs = run_for_one()
    # for sign in detected_signs:
    #     print(sign)

    # texts, cm = run_for_many(2, random_pick=False)

    # for t in texts:
    #     print(t)

    # print("\nMACIERZ POMYŁEK:")
    # for true_lbl, preds in cm.items():
    #     for pred_lbl, count in preds.items():
    #         print(f"{true_lbl} -> {pred_lbl}: {count}")

    # calculate_and_plot_summary_metrics(cm)

    # image_name = "0000036.jpg"
    # categories = get_true_label( image_name)
    # print(f"Kategorie na obrazie {image_name}:")
    # for c in categories:
    #     print("-", c)
