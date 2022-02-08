from skimage.feature import greycomatrix, greycoprops
from joblib import delayed, Parallel
from torch.nn import Module
import numpy as np
import torch

@torch.no_grad()
class Backbone(Module):
    STAT_AXIS = (-2, -1)
    GLCM_PROPERTIES = ('contrast', 'homogeneity', 'energy', 'correlation')

    def __init__(self, ndvi_mode, ndvi_stat, space_mode, bins_path, dist, theta, n_jobs=4):
        super().__init__()
        self.training = False

        self.ndvi_mode = ndvi_mode
        self.mean = ndvi_stat.__dict__[f'{ndvi_mode}_mean']
        self.std = ndvi_stat.__dict__[f'{ndvi_mode}_std']
        
        self.space_mode = space_mode
        self.stride = [4, 4] if space_mode == 'local' else [1, 1]
        self.kernel_size = [17, 17] if space_mode == 'local' else [300, 190]

        self.bins_path = bins_path
        self.bins = np.load(bins_path)
        self.dist = dist
        self.theta = np.array(theta) * np.pi
        self.n_jobs = n_jobs

    @property
    def features_len(self): 
        dummy_input = np.random.rand(1, 1, 300, 190) * 2 - 1 # for ndvi [-1, 1]
        return self.forward(dummy_input).shape[-1]

    def get_feature_slice(self, group):
        stat_len = 4
        glcm_len = len(self.theta) * len(self.dist) * 5
        if group == 'ALL':
            return slice(None)
        if group == 'STAT':
            return slice(None, stat_len)
        if group == 'HIST':
            return slice(stat_len, -glcm_len)
        if group == 'GLCM':
            return slice(-glcm_len, None)


    def unfold(self, x): 
        x = torch.Tensor(x)
        batch_size = x.shape[0]
        x = x.unfold(2, self.kernel_size[0], self.stride[0])
        x = x.unfold(3, self.kernel_size[1], self.stride[1])
        x = x.reshape(batch_size, 1, -1, self.kernel_size[0], self.kernel_size[1])
        return x.numpy()

    def statistics(self, imgs):
        mean = imgs.mean(axis=self.STAT_AXIS)
        std = imgs.std(axis=self.STAT_AXIS)
        max = (imgs.max(axis=self.STAT_AXIS) - mean) / std
        min = (mean - imgs.min(axis=self.STAT_AXIS)) / std
        return np.concatenate([mean, std, max, min], axis=1)

    def hist(self, q_img):
        q_img = q_img.flatten()
        if len(set(q_img)) > 1:
            hist, _ = np.histogram(q_img, bins=np.arange(len(self.bins) + 2), density=True)
        else:
            hist = np.zeros(len(self.bins) + 1)
        return hist

    def glcm(self, q_img):
        levels = len(set(self.bins)) + 1
        g = greycomatrix(q_img, self.dist, self.theta, levels, normed=True, symmetric=True)
        props = np.array([greycoprops(g, p) for p in self.GLCM_PROPERTIES]).reshape(-1)
        entropy = -np.sum(np.multiply(g, np.log2(g + 1e-8)), axis=(0, 1)).reshape(-1)
        props = np.concatenate([props, entropy], axis=0)
        return props
    
    def quantize_hist_glcm(self, img):
        img = (img - self.mean) / self.std
        q_img = np.digitize(img, self.bins)
        hist = self.hist(q_img)
        glcm = self.glcm(q_img)
        return np.concatenate([hist, glcm])
    
    def run_parallel(self, imgs, function):
        shape = imgs.shape
        windows = imgs.reshape(-1, shape[-2], shape[-1])
        windows_data = Parallel(n_jobs=self.n_jobs)(delayed(function)(img) for img in windows)
        windows_data = np.stack(windows_data, axis=0)
        windows_data = windows_data.reshape(shape[0], shape[2], -1)
        windows_data = np.transpose(windows_data, (0, 2, 1))
        return windows_data

    def forward(self, x):                                         # B, C, H, W
        x = self.unfold(x)                                        # B, C, n_windows, krenel_h, krenel_w
        stat = self.statistics(x)                                 # B, [STAT], n_windows
        hist_glcm = self.run_parallel(x, self.quantize_hist_glcm) # B, [HIST, GLCM], n_windows
        x = np.concatenate([stat, hist_glcm], axis=1)             # B, [STAT, HIST, GLCM], n_windows
        x = np.mean(x, axis=(-1), keepdims=False)                 # B, [STAT, HIST, GLCM]
        return x