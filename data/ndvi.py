import os
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.dataset import WheatDS
from data.transforms import NDVI, MEAN_LEFT_NDVI_T, MEAN_LEFT_NDVI_G, STD_NDVI_T, STD_NDVI_G


if __name__ == '__main__':
    src_root = os.path.join(os.getcwd(), 'ds/struct_png_clear/box_IR_90')
    dst_root = os.path.join(os.getcwd(), 'ds/ndvi_whiting/box_IR_90')
    if os.path.exists(dst_root):
        print(f'Folder {dst_root} already exists')
    else:
        os.makedirs(dst_root)
        dst_tir_root, dst_rgb_root = dst_root + '\\tir', dst_root + '\\rgb'
        os.makedirs(dst_tir_root)
        os.makedirs(dst_rgb_root)

        transform = transforms.Compose([
            NDVI(),
        ])

        DS = WheatDS(src_root, transform)
        loader = DataLoader(DS)

        for sample in tqdm(loader):
            ndvi_t, ndvi_g, name, label = sample
            ndvi_t = ndvi_t.numpy()[0]
            ndvi_g = ndvi_g.numpy()[0]

            ndvi_t = (ndvi_t - MEAN_LEFT_NDVI_T) / STD_NDVI_T
            ndvi_g = (ndvi_g - MEAN_LEFT_NDVI_G) / STD_NDVI_G
            ndvi_t = (ndvi_t + 3) / 6
            ndvi_g = (ndvi_g + 3) / 6

            cv2.imwrite(dst_tir_root + '\\' + ''.join(name), ndvi_t * 256)
            cv2.imwrite(dst_rgb_root + '\\' + ''.join(name), ndvi_g * 256)