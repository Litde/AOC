import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from tqdm import tqdm

class SignDataset:
    def __init__(self, data_path, label, img_size=(64, 64)):
        self.data_path = data_path
        self.label = label
        self.img_size = img_size
        self.X = []
        self.y = []
        self.data = self.load_data()

    def load_data(self):
        self.X = []
        self.y = []
        for root, _, files in tqdm(os.walk(self.data_path), desc=f'Loading {self.label} data'):
            for img_file in files:
                if img_file.startswith('.'):
                    continue
                img_path = os.path.join(root, img_file)
                try:
                    img = imread(img_path, as_gray=True)
                except Exception:
                    continue
                img_resized = resize(img, self.img_size).flatten()
                self.X.append(img_resized)
                self.y.append(self.label)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        return self.X

    def preprocess_data(self):
        print("Preprocessing data")
        pass

    def get_data(self, idx):
        return self.data[idx]

class MultiSignDataset:
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.circle = None
        self.triangle = None
        self.rectangle = None
        self.square = None

    def load(self):
        self.circle = SignDataset('enhanced_input/circle', 'circle', self.img_size)
        self.triangle = SignDataset('enhanced_input/triangle', 'triangle', self.img_size)
        self.rectangle = SignDataset('enhanced_input/rectangle', 'rectangle', self.img_size)
        self.square = SignDataset('enhanced_input/square', 'square', self.img_size)

        print("Circle samples:", len(self.circle.y))
        print("Triangle samples:", len(self.triangle.y))
        print("Rectangle samples:", len(self.rectangle.y))
        print("Square samples:", len(self.square.y))
