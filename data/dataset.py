import os
import cv2
from torch.utils.data import Dataset


class WheatDS(Dataset):
    def __init__(self, root, transform):
        self.data = self.load_data(root)
        self.transform = transform

    @staticmethod
    def load_data(root):
        tir_root = os.path.join(root, 'tir')
        tir_paths = [os.path.join(tir_root, img) for img in os.listdir(tir_root)]
        rgb_root = os.path.join(root, 'rgb')
        rgb_paths = [os.path.join(rgb_root, img) for img in os.listdir(rgb_root)]
        data = {'tir': tir_paths, 'rgb': rgb_paths}
        return data

    def get_data(self):
        return self.data

    def __len__(self):
        return len(self.data['tir'])

    def __getitem__(self, item):
        tir_path = self.data['tir'][item]
        rgb_path = self.data['rgb'][item]
        name = tir_path.split('\\')[-1]
        tir = cv2.imread(tir_path)
        rgb = cv2.imread(rgb_path)
        if self.transform != None:
            tir, rgb, name = self.transform((tir, rgb, name))
        return tir, rgb, name