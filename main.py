from ClassificationModel import ClassificationRandomForest
from ClusterSVM import ClusterSVM


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

def test_svm():
    model = ClusterSVM(target_class='triangle', img_size=(64, 64))
    model.load_model('models/binary_triangle_model.joblib')

    y_pred = model.predict('input/rectangle/I-38/001_201311251210402_5901_celje.png')
    print("Predicted label:", y_pred)

def test_rf():
    model = ClassificationRandomForest(data_dir='input/circle', img_size=(64, 64))
    model.load_model('models/rf_circle_model.joblib')

    y_pred = model.predict('input/circle/II-3/001_201311071110100_13511_celje.png')
    print("Predicted label:", y_pred)

if __name__ == "__main__":
    # train_svm()
    train_rf()
    # test_rf()
