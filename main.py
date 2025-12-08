from ClassificationModel import ClassificationRandomForest
from ClusterSVM import ClusterSVM

import detectors.byHough as hough
import detectors.byApproxPolyDP as approxPolyDP
import detectors.bySegmentation as segmentation

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
    model = ClusterSVM(target_class='triangle', img_size=(64, 64))
    model.load_model(model_path)

    return model.predict(image_path)

def test_rf(image_path, shape):
    model_path = f"models/rf_{shape}_model.joblib"
    model = ClassificationRandomForest(data_dir='cropped/', img_size=(64, 64))
    model.load_model(model_path)

    return model.predict(image_path)


def main():
    image_path = "JPEGImages/0000289.jpg"
    all_crops = []

    modules  = [
        ("hough", hough),
        ("approx", approxPolyDP),
        ("segmentation", segmentation)
    ]

    for name, mod in modules:
        print(f"Running {name}...")
        try:
            crops = mod.run_detector(image_path)
            all_crops.extend(crops)
        except Exception as e:
            print(f"Error in {name}:", e)

    print("\nDETECTED SIGNS: ")
    printed_labels = set()

    for filename in all_crops:

        if "circle" in filename or "octagon" in filename:
            shape = "circle"
        elif "triangle" in filename:
            shape = "triangle"
        elif "square" in filename or "rectangle" in filename:
            shape = "rectangle"
        else:
            shape = "unknown"

        if shape == "unknown":
            continue

        label = test_rf(filename, shape)

        if label.lower() == "X-1.2" or label in printed_labels:
            continue

        print(label)
        printed_labels.add(label)





if __name__ == "__main__":
    # train_rf()
    # train_svm()

    # test_svm()
    # test_rf()

    main()
