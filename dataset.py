import numpy as np
import transforms
import torch
import json
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, augments=None):
        with open(data_json) as f:
            self.data = json.load(f)

        # self.transforms = augments
        self.transforms = None
        if augments != None:
            self.transforms = []
            for aug in augments:
                aug_name = list(aug.__dict__.keys())[0]
                aug_class = transforms.__getattribute__(aug_name)
                aug = aug_class(**aug.__dict__[aug_name].__dict__)
                self.transforms.append(aug)
            self.transforms = transforms.ComposeTransform(self.transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        tir = cv2.imread(self.data[id]['tir'], cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(self.data[id]['rgb'])
        reg_label = self.data[id]['reg_label']
        clas_label = self.data[id]['clas_label']
        name = self.data[id]['tir'].split('/')[-1]

        if self.transforms:
            tir, rgb, name = self.transforms(tir, rgb, name)

        tir = np.asarray(tir, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.float32)
        reg_label = np.asarray(reg_label, dtype=np.float32)
        clas_label = np.asarray(clas_label, dtype=np.float32)

        red   = rgb[:, :, 2]
        green = rgb[:, :, 1]

        ndvi_t = (tir   - red) / (tir   + red + 1e-17)
        ndvi_g = (green - red) / (green + red + 1e-17)

        ndvi_t = np.expand_dims(ndvi_t, axis=0)
        ndvi_g = np.expand_dims(ndvi_g, axis=0)
        reg_label = np.expand_dims(reg_label, axis=-1)
        clas_label = np.expand_dims(reg_label, axis=-1)

        return ndvi_t, ndvi_g, reg_label, clas_label, name

class FeachDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, feature_slice):
        self.data = data
        self.data = self.data[:, feature_slice]
        self.labels = labels
        
    @property
    def feach_len(self):
        return self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id], np.array([self.labels[id],])

