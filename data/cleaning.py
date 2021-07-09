import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.dataset import WheatDS


class Normalize(object):
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


class RenameRGB(object):
    def __init__(self, rgb_root, before, after):
        self.root = rgb_root
        self.before = before
        self.after = after

    def __call__(self, sample):
        tir, rgb, name = sample
        src = os.path.join(self.root, name)
        name = name.replace(self.before, self.after)
        dst = os.path.join(self.root, name)
        os.rename(src, dst)
        return tir, rgb, name


if __name__ == '__main__':
    src_root = os.path.join(os.getcwd(), 'ds/struct_png/box_IR_90')
    dst_root = os.path.join(os.getcwd(), 'ds/struct_png_clear/box_IR_90')
    if os.path.exists(dst_root):
        print(f'Folder {dst_root} already exists')
    else:
        os.makedirs(dst_root)
        dst_tir_root, dst_rgb_root = dst_root + '\\tir', dst_root + '\\rgb'
        os.makedirs(dst_tir_root)
        os.makedirs(dst_rgb_root)

        transform = transforms.Compose([
            Normalize(),
            RenameRGB(src_root + '\\rgb', '_Реальное изображение', ''),
        ])

        DS = WheatDS(src_root, transform)
        loader = DataLoader(DS)
        for sample in tqdm(loader):
            tir, rgb, name = sample
            cv2.imwrite(dst_tir_root + "\\" + ''.join(name), tir.numpy()[0])
            cv2.imwrite(dst_rgb_root + "\\" + ''.join(name), rgb.numpy()[0])
        print(f'Create {dst_root}')

