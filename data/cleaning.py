import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.dataset import WheatDS
from data.transforms import Rename, Fusion


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
            Fusion(),
            Rename(src_root + '\\rgb', '_Реальное изображение', ''),
        ])

        DS = WheatDS(src_root, transform)
        loader = DataLoader(DS)
        for sample in tqdm(loader):
            tir, rgb, name = sample
            cv2.imwrite(dst_tir_root + "\\" + ''.join(name), tir.numpy()[0])
            cv2.imwrite(dst_rgb_root + "\\" + ''.join(name), rgb.numpy()[0])
        print(f'Create {dst_root}')
