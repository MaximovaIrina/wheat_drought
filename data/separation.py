import os
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.dataset import WheatDS


class Separate(object):
    def __call__(self, sample):
        tir, rgb, name = sample
        h, w, _ = tir.shape
        tir_l, tir_r = tir[:, :w // 2 - 10, :], tir[:, w // 2 + 10:, :]
        rgb_l, rgb_r = rgb[:, :w // 2 - 10, :], rgb[:, w // 2 + 10:, :]
        return (tir_l, tir_r), (rgb_l, rgb_r), name


if __name__ == '__main__':
    src_root = os.path.join(os.getcwd(), 'ds/struct_png_clear/box_IR_90')
    dst_root = os.path.join(os.getcwd(), 'ds/struct_png_clear_separate/box_IR_90')
    if os.path.exists(dst_root):
        print(f'Folder {dst_root} already exists')
    else:
        os.makedirs(dst_root)
        dst_tir_root, dst_rgb_root = dst_root + '\\tir', dst_root + '\\rgb'
        os.makedirs(dst_tir_root)
        os.makedirs(dst_rgb_root)

        transform = transforms.Compose([
            Separate(),
        ])

        DS = WheatDS(src_root, transform)
        loader = DataLoader(DS)
        for sample in tqdm(loader):
            tir, rgb, name = sample
            tir_l, tir_r = tir
            rgb_l, rgb_r = rgb
            cv2.imwrite(dst_tir_root + "\\l_" + ''.join(name), tir_l.numpy()[0])
            cv2.imwrite(dst_tir_root + "\\r_" + ''.join(name), tir_r.numpy()[0])
            cv2.imwrite(dst_rgb_root + "\\l_" + ''.join(name), rgb_l.numpy()[0])
            cv2.imwrite(dst_rgb_root + "\\r_" + ''.join(name), rgb_r.numpy()[0])
        print(f'Create {dst_root}')