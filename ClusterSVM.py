from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from dataset import MultiSignDataset



class ClusterSVM:
    def __init__(self, target_class, img_size:tuple=(64, 64)):
        self.target_class = target_class
        self.img_size = img_size
        self.multi_ds = MultiSignDataset(img_size)
        self.X = []
        self.y = []
        self.model = None

    def load_data(self):
        self.multi_ds.load()
        self.X = np.concatenate([self.multi_ds.circle.X, self.multi_ds.triangle.X, self.multi_ds.rectangle.X])
        y_combined = np.concatenate([self.multi_ds.circle.y, self.multi_ds.triangle.y, self.multi_ds.rectangle.y])
        self.y = np.where(y_combined == self.target_class, 1, 0)

    def train_model(self, param_grid:dict):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        best_score = -np.inf
        best_params = None
        param_list = list(ParameterGrid(param_grid))
        for params in tqdm(param_list, desc='GridSearch'):
            model = svm.SVC(**params)
            try:
                scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=4)
            except Exception as e:
                print(f"Skipping params:{params}, Exception{e}")
                continue
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        if best_params is None:
            raise RuntimeError('Grid search failed to find any valid parameter configuration')

        self.model = svm.SVC(**best_params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, img_path):
        X = self.prepare_input(img_path)
        prediction = self.model.predict(X)
        if prediction.shape[0] == 1:
            return self.target_class if prediction[0] == 1 else f'not_{self.target_class}'
        return [self.target_class if p == 1 else f'not_{self.target_class}' for p in prediction]

    def prepare_input(self, imgs):
        def process_single(item):
            if isinstance(item, str):
                try:
                    arr = imread(item, as_gray=True)
                except Exception as e:
                    raise ValueError(f'Failed to read image from path {item}: {e}')
            else:
                arr = np.array(item)

            if arr.ndim == 3:
                try:
                    from skimage.color import rgb2gray
                    arr = rgb2gray(arr)
                except Exception:
                    arr = arr.mean(axis=2)

            if arr.ndim != 2:
                raise ValueError(f'Unsupported image array shape: {arr.shape}')

            arr_resized = resize(arr, self.img_size).flatten().astype(np.float32)
            return arr_resized

        items = None
        if isinstance(imgs, (list, tuple)):
            items = imgs
        elif isinstance(imgs, np.ndarray):
            if imgs.ndim == 4:
                items = [imgs[i] for i in range(imgs.shape[0])]
            else:
                items = [imgs]
        else:
            items = [imgs]

        processed = []
        for it in items:
            processed.append(process_single(it))

        X = np.vstack(processed).reshape(len(processed), -1)
        return X

    def save_model(self, model_path):
        import joblib
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        import joblib
        self.model = joblib.load(model_path)
