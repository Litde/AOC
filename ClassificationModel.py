import os
import numpy as np
from skimage import io, color
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class ClassificationRandomForest:
    def __init__(self, data_dir, img_size=(64, 64), n_estimators=100, max_depth=None, random_state=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.X = []
        self.y = []

    def load_data(self) -> None:
        self.X = []
        self.y = []
        for class_name in tqdm(os.listdir(self.data_dir), desc='Loading data'):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.startswith('.'):
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        img = io.imread(file_path, as_gray=True)
                        img_resized = resize(img, self.img_size, anti_aliasing=True)
                        self.X.append(img_resized.flatten().astype(np.float32))
                        self.y.append(class_name)
                    except Exception:
                        continue
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def prepare_input(self, imgs) -> np.ndarray:
        if isinstance(imgs, str):
            imgs = [imgs]
        elif isinstance(imgs, np.ndarray) and imgs.ndim == 2:
            imgs = [imgs]
        elif isinstance(imgs, np.ndarray) and imgs.ndim == 3:
            imgs = [imgs]
        elif isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            pass
        else:
            imgs = list(imgs)

        processed = []
        for img in imgs:
            if isinstance(img, str):
                img = io.imread(img, as_gray=True)
            elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                img = color.rgb2gray(img)
            img_resized = resize(img, self.img_size, anti_aliasing=True)
            processed.append(img_resized.flatten().astype(np.float32))
        return np.array(processed)

    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, imgs) -> np.ndarray:
        X = self.prepare_input(imgs)
        predictions = self.model.predict(X)
        return predictions[0] if X.shape[0] == 1 else predictions

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro')
        }
        return metrics

    def train_model(self) -> dict:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.fit(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        return metrics

    def save_model(self, model_path) -> None:
        import joblib
        joblib.dump(self.model, model_path)

    def load_model(self, model_path) -> None:
        import joblib
        self.model = joblib.load(model_path)
