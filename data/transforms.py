import numpy as np
import cv2
import os

MEAN_LEFT_NDVI_T = 0.137
MEAN_LEFT_NDVI_G = 0.072
STD_NDVI_T = 0.321
STD_NDVI_G = 0.092


class NDVI(object):
    def __call__(self, sample):
        tir, rgb, name = sample
        tir = cv2.cvtColor(tir, cv2.COLOR_BGR2GRAY)
        red = rgb[:, :, 2]
        green = rgb[:, :, 1]

        tir = np.asarray(tir, dtype=np.float32)
        red = np.asarray(red, dtype=np.float32)
        green = np.asarray(green, dtype=np.float32)

        ndvi_t = (tir - red) / (tir + red + 1e-7)
        ndvi_g = (green - red) / (green + red + 1e-7)

        ndvi_t += (MEAN_LEFT_NDVI_T - np.mean(ndvi_t))
        ndvi_g += (MEAN_LEFT_NDVI_G - np.mean(ndvi_g))

        name_attr = name.split('.')[0].split('_')
        pos, day = name_attr[0], int(name_attr[-1])
        label = day if pos == 'r' else 0
        return ndvi_t, ndvi_g, name, label


class Rename(object):
    def __init__(self, root, before, after):
        self.root = root
        self.before = before
        self.after = after

    def __call__(self, sample):
        tir, rgb, name = sample
        src = os.path.join(self.root, name)
        if self.before in name:
            name = name.replace(self.before, self.after)
            dst = os.path.join(self.root, name)
            os.rename(src, dst)
        return tir, rgb, name


class Fusion(object):
    def __init__(self, coef=0.56, bias_h=50, bias_w=10):
        self.coef = coef
        self.bias_h = bias_h
        self.bias_w = bias_w

    def __call__(self, sample):
        tir, rgb, name = sample
        h, w, _ = rgb.shape
        if h == 300 and w == 400:
            return rgb, tir

        tir = tir[:, :1300, ]
        tir = cv2.resize(tir, (int(w * self.coef), int(h * self.coef)))

        h_, w_, _ = tir.shape
        rgb = rgb[(h - h_) // 2 + self.bias_h: (h + h_) // 2 + self.bias_h,
                  (w - w_) // 2 + self.bias_w: (w + w_) // 2 + self.bias_w, ]

        tir = tir[100:h_ - 100, 100: w_ - 150, :]
        rgb = rgb[100:h_ - 100, 100: w_ - 150, :]

        tir = cv2.resize(tir, (400, 300))
        rgb = cv2.resize(rgb, (400, 300))
        return (tir, rgb, name)


class Separate(object):
    def __call__(self, sample):
        tir, rgb, name = sample
        h, w, _ = tir.shape
        tir_l, tir_r = tir[:, :w // 2 - 10, :], tir[:, w // 2 + 10:, :]
        rgb_l, rgb_r = rgb[:, :w // 2 - 10, :], rgb[:, w // 2 + 10:, :]
        return (tir_l, tir_r), (rgb_l, rgb_r), name