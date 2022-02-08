import cv2


class Calibration:
    def __init__(self, ratio=0.56, dh=50, dw=10):
        self.ratio = ratio  # Approximate box size ratio - RGB/TIR
        self.dh = dh        # Offset of box centres vertically
        self.dw = dw        # Offset of box centres horizontally

    def __call__(self, tir, rgb, name):
        h, w, = rgb.shape[:2]
        h_, w_ = int(h * self.ratio), int(w * self.ratio)
        
        tir = tir[:, :1300, ]
        tir = cv2.resize(tir, (w_, h_))

        rgb = rgb[(h-h_)//2 + self.dh : (h+h_)//2 + self.dh, 
                  (w-w_)//2 + self.dw : (w+w_)//2 + self.dw]

        tir = tir[100 : h_-100, 100 : w_-150]
        rgb = rgb[100 : h_-100, 100 : w_-150]

        tir = cv2.resize(tir, (400, 300))
        rgb = cv2.resize(rgb, (400, 300))
        return tir, rgb, name


class ExtractRightHalf:
    def __init__(self, offset=10):
        self.offset = offset

    def __call__(self, tir, rgb, name):
        h, w = tir.shape
        tir = tir[:, w//2 + self.offset:]
        rgb = rgb[:, w//2 + self.offset:]
        return tir, rgb, name


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tir, rgb, name):
        for t in self.transforms:
            tir, rgb, name = t(tir, rgb, name)
        return tir, rgb, name