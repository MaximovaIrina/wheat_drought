from skimage.filters import threshold_multiotsu, threshold_otsu
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from transforms import *

class WhiteningNDVI:
    def __init__(self):
        self.mean_ndvi_t = -0.125
        self.std_ndvi_t = 0.437
        self.mean_ndvi_g = 0.060
        self.std_ndvi_g = 0.070

    def __call__(self, ndvi_t, ndvi_g, name):
        ndvi_t = (ndvi_t - self.mean_ndvi_t) / self.std_ndvi_t
        ndvi_g = (ndvi_g - self.mean_ndvi_g) / self.std_ndvi_g
        return ndvi_t, ndvi_g, name


class HistQuantStatic:
    def __init__(self, classes, **kwargs):
        super().__init__()
        assert classes % 2 == 0
        self.classes = classes
        classes -= 2 # границные класссы
        self.bins = np.arange(-classes/2, classes/2 + 1, step=1)

    def thresholds(self, samples):
        thresholds = np.zeros(self.bins.shape) + np.asarray(self.bins)
        assert len(thresholds) == self.classes - 1
        return thresholds


class HistQuantMultiOtsu:
    def __init__(self, classes, nbins=50, **kwargs):
        self.classes = classes
        self.nbins = nbins

    def thresholds(self, samples):
        thresholds =  threshold_multiotsu(samples, self.classes, self.nbins)
        assert len(thresholds) == self.classes - 1
        return  thresholds


class HistQuantLocalMins:
    def __init__(self, classes, logging_root=None, **kwargs):
        self.classes = classes
        self.logging_root = logging_root
        if self.logging_root:
            self.logging_root += '/HistQuantLocalMins'
            os.makedirs(self.logging_root, exist_ok=True)

    def _plot_derivative(self, bins, hist, hist_d1, thresholds):
        plt.plot(bins, hist, color='red', label='src_hist')
        plt.plot(bins[:-1], hist_d1, color='blue', alpha=0.5, label='1st derivative')
        plt.hlines(0, xmin=-3, xmax=3, color='black',  alpha=0.5)
        ymin, ymax = plt.ylim()
        for min in thresholds:
            plt.vlines(min, ymin=ymin, ymax=ymax, color='gray', alpha=0.5)
        plt.legend()
        plt.ylabel('probability')
        plt.xlabel('NDVI_T')
        plt.savefig(self.logging_root + '/hist_d1.png', dpi=300)
        plt.close()

    def thresholds(self, samples):
        hist = sns.histplot([-3,] + list(samples.flatten()) + [3,], kde=True, stat='probability')
        bins, hist = hist.get_lines()[0].get_data()
        plt.close()

        hist_d1 = np.diff(hist)/np.diff(bins)
        thresholds = list(bins[np.where(np.diff(np.sign(hist_d1)) > 0)]) # Точки минимума по нулю производной
        if len(thresholds) != self.classes - 1:
            inflections = []
            for i in range(1, len(hist_d1) - 1): # точки локального максимума отрицательной производной (точки перегиба в убывании) 
                if np.all(hist_d1[i-1: i+2] <= 0) and np.max(hist_d1[i-1: i+2]) == hist_d1[i]:
                    inflections.append(bins[i])
            np.random.seed(0)
            radom_ids = np.random.choice(np.arange(len(inflections)), size=(self.classes - 1 - len(thresholds),))
            inflections = np.array(inflections)[radom_ids]
            thresholds += list(inflections)

        multi_otsu_thrs = HistQuantMultiOtsu(self.classes).thresholds(samples)
        multi_otsu_thrs = np.array(multi_otsu_thrs)
        thresholds = np.array(thresholds)
        if len(thresholds) == 0:
            thresholds = multi_otsu_thrs
        elif len(thresholds) != self.classes - 1: # берем ту что в среднем наиболее удаленная  от найденных
            diffs = np.zeros((len(multi_otsu_thrs), ))
            for i, mot in enumerate(multi_otsu_thrs):
                diffs[i] = np.abs(mot - thresholds).mean()
            thresholds = list(thresholds)
            thresholds += mot[np.argsort(diffs)[::-1][:self.classes - 1 - len(thresholds)]]

        
        assert len(thresholds) == self.classes - 1
        if self.logging_root:
            self._plot_derivative(bins, hist, hist_d1, thresholds)
        return thresholds


class HistQuantMRI:
    def __init__(self, classes, epsilon=0.1, multiotsu_nbins=50, binotsu_nbins=10, logging_root=None, **kwargs):
        self.classes = classes
        self.epsilon = epsilon
        self.multiotsu_nbins = multiotsu_nbins
        self.binotsu_nbins = binotsu_nbins
        self.logging_root = logging_root
        if self.logging_root:
            self.logging_root += '/HistQuantMRI'
            os.makedirs(self.logging_root, exist_ok=True)

    def _plot_iteration(self, store, means_borders=None):
        hist = sns.histplot([-3,] + list(store['samples'].flatten()) + [3,], kde=True, stat='probability')
        bins, hist = hist.get_lines()[0].get_data()
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(bins, hist, color='red')
        ymin, ymax = plt.ylim()
        for t in store['thresholds'][-1]:
            ax.axvline(x=t, c='blue', lw=1)
        if not isinstance(means_borders, type(None)):
            for x1, x2 in means_borders:
                rect = matplotlib.patches.Rectangle((x1, ymin), x2-x1, ymax-ymin, color='yellow', alpha=0.5)
                ax.axvline(x=x1, c='grey', lw=0.2)
                ax.axvline(x=x2, c='grey', lw=0.2)
                ax.add_patch(rect)
        it = len(store['thresholds'])
        ax.set_title(f'{it}')
        fig.savefig(self.logging_root + f'/{it}it.png', dpi=300)
        plt.close()

    def _plot_store(self, store):
        hist = sns.histplot([-3,] + list(store['samples'].flatten()) + [3,], kde=True, stat='probability')
        bins, hist = hist.get_lines()[0].get_data()
        plt.close()
        plt.plot(bins, hist, color='red')
        
        cmap = matplotlib.cm.get_cmap('cool_r')
        color_id = np.linspace(0, 1, len(store['thresholds']))
        for it, thrs in enumerate(store['thresholds']):
            color = cmap(color_id[it])
            plt.axvline(x=thrs[0], c=color, label=f'{it+1}it')
            for thr in thrs[1:]:
                plt.axvline(x=thr, c=color)

        plt.legend()
        plt.savefig(self.logging_root + '/dymamic.png', dpi=300)
        plt.close()

    def thresholds(self, samples):
        thresholds = threshold_multiotsu(samples, self.classes, self.multiotsu_nbins)
        store = {'samples': samples, 'thresholds': [thresholds]}
        if self.logging_root:
            self._plot_iteration(store)

        full_thresholds = [-3, ] + list(thresholds) + [3, ]
        means_borders = np.zeros((len(thresholds), 2))
        for i in range(len(thresholds)):
            means_borders[i] = np.array([full_thresholds[i], full_thresholds[i+2]])
        
        while not np.all(means_borders[:,1] - means_borders[:,0] < self.epsilon):
            for i, borders in enumerate(means_borders):
                l, r = borders
                if r - l > self.epsilon:
                    l = samples[np.where((l             < samples) & (samples < thresholds[i]))].mean()
                    r = samples[np.where((thresholds[i] < samples) & (samples < r            ))].mean()
                    data = samples[np.where((l < samples) & (samples < r))]
                    thresholds[i] = threshold_otsu(data, self.binotsu_nbins)
                    means_borders[i] = np.array([l, r])       
            store['thresholds'] = np.vstack([store['thresholds'], thresholds])
            if self.logging_root: 
                self._plot_iteration(store, means_borders)
        
        if self.logging_root:
            self._plot_store(store)
        assert len(thresholds) == self.classes - 1
        return thresholds


class HistQuantAnalysis:
    def __init__(self, box_id='box3', logging_root=None, **kwargs):
        self.ndvi_t = []
        self.ndvi_g = []
        self.names = []
        self.box_id = box_id

        self.logging_root = logging_root
        if self.logging_root:
            self.logging_root += '/HistQuantAnalysis'
            os.makedirs(self.logging_root, exist_ok=True)
    
    def plot_common_histogram(self):
        sort = np.argsort(self.names)
        self.names = np.asarray(self.names)[sort]
        self.ndvi_t = np.asarray(self.ndvi_t)[sort]
        self.ndvi_g = np.asarray(self.ndvi_g)[sort]
        cmap = sns.color_palette("flare_r", n_colors=len(self.names), as_cmap=False)

        for name, ndvi in [('NDVI_T', self.ndvi_t), ('NDVI_G', self.ndvi_g)]:
            xs, ys = [], []
            for i in range(len(self.names)):
                x = ndvi[i].flatten()
                x = x[np.where(x>-2)]
                x = [-3,] + list(x) + [3,]
                hist = sns.histplot(x, kde=True, stat='probability')
                x_data, y_data = hist.get_lines()[0].get_data()
                xs.append(x_data)
                ys.append(y_data)
                plt.close()

            if name == 'NDVI_G':
                xs[-1] += 0.5

            for i in range(len(self.names)):
                plt.plot(xs[i], ys[i], label=self.names[i], color=cmap[i])
            plt.legend(title="day")
            plt.xlim(-3, 3)
            plt.xticks(np.arange(-3, 4, 1), ['-3σ', '-2σ', '-σ', 'μ', 'σ', '2σ', '3σ'])
            plt.xlabel(name)
            plt.ylabel('probability')
            plt.savefig(self.logging_root + f'/{self.box_id}_common_histogram_{name}.png', dpi=300)
            plt.close()

    def merge_distribution(self, healthy, drought):
        shape = healthy.shape
        h_hist = sns.histplot([-3,] + list(healthy.flatten()) + [3,], kde=True, stat='probability')
        d_hist = sns.histplot([-3,] + list(drought.flatten()) + [3,], kde=True, stat='probability')
        h_bibs, h_hist = h_hist.get_lines()[0].get_data()
        d_bibs, d_hist = d_hist.get_lines()[1].get_data()
        plt.close()
        
        hist = np.maximum(h_hist, d_hist)
        hist /=  np.sum(hist)

        generator = sps.rv_discrete(values=(range(len(hist)), hist), seed=0)  # значения и вероятности
        idx = generator.rvs(size=len(healthy.flatten()))
        samples = h_bibs[idx]
        return samples.reshape(shape)
    
    def plot_thresholds(self, healthy, drought, samples, thr, days, method_name):
        h_hist = sns.histplot([-3,] + list(healthy.flatten()) + [3,], kde=True, stat='probability')
        d_hist = sns.histplot([-3,] + list(drought.flatten()) + [3,], kde=True, stat='probability')
        merge_hist = sns.histplot([-3,] + list(samples.flatten()) + [3,], kde=True, stat='probability')
        h_bins, h_hist = h_hist.get_lines()[0].get_data()
        d_bins, d_hist = d_hist.get_lines()[1].get_data()
        merge_bins, merge_hist = merge_hist.get_lines()[2].get_data()
        plt.close()

        fig, axs = plt.subplots(2, 2, figsize=(5, 8))
        axs[0][0].imshow(healthy, cmap='gray')
        axs[0][0].set_title(f'src NDVI_T, day {days[0]}')
        axs[0][1].imshow(drought, cmap='gray')
        axs[0][1].set_title(f'src NDVI_T, day {days[1]}')
        axs[1][0].imshow(np.digitize(healthy, bins=thr), cmap='gray')
        axs[1][0].set_title(f'quant NDVI_T, day {days[0]}')
        axs[1][1].imshow(np.digitize(drought, bins=thr), cmap='gray')
        axs[1][1].set_title(f'quant NDVI_T, day {days[1]}')
        [ax.axis('off') for ax in axs.flatten()]
        fig.canvas.draw()
        images = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()

        fig, axs = plt.subplots(3, 1, figsize=(5, 8))
        axs[0].plot(h_bins, h_hist, color='g', alpha=0.5, label='healthy')
        axs[0].plot(h_bins, d_hist, color='b', alpha=0.5, label='drought')
        axs[1].plot(h_bins, merge_hist, color='r', label='merge')
        thr = [-3, ] + list(thr) + [3, ]
        for t in thr:
            axs[1].axvline(x=t, c='grey', lw=1)
        axs[2].hist(healthy.flatten(), thr, color='g', alpha=0.5, label='healthy', density=True)
        axs[2].hist(drought.flatten(), thr, color='b', alpha=0.5, label='drought', density=True)
        axs[2].set_xticks(np.arange(-3, 4, 1), ['-3σ', '-2σ', '-σ', 'μ', 'σ', '2σ', '3σ'])
        [ax.legend(loc='upper right') for ax in axs.flatten()]
        axs[0].xaxis.set_visible(False)
        axs[1].xaxis.set_visible(False)
        fig.canvas.draw()
        hists = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()

        plot = np.concatenate([hists, images], 1)
        plt.imshow(plot)
        plt.axis('off')
        plt.savefig(self.logging_root + f'/{self.box_id}_method_{method_name}.png', dpi=300, bbox_inches='tight', pad_inches = 0)
        plt.close()

    def quantize_analysis(self, method, days):
        all_days = np.asarray([int(name) for name in self.names])
        healthy = np.asarray(self.ndvi_t)[np.where(all_days == days[0])][0]
        drought = np.asarray(self.ndvi_t)[np.where(all_days == days[1])][0]
        samples = self.merge_distribution(healthy, drought)
        thresholds = method.thresholds(samples)
        method_name = str(method.__class__).split('\'')[1].split('.')[-1]
        np.save(self.logging_root + f'/{method_name}', thresholds)
        self.plot_thresholds(healthy, drought, samples, thresholds, days, method_name)

    def __call__(self, ndvi_t, ndvi_g, name):
        if self.box_id in name:
            self.ndvi_t.append(ndvi_t)
            self.ndvi_g.append(ndvi_g)
            self.names.append(name.split('.')[0].split('_')[-1])
        return ndvi_t, ndvi_g, name


import dataset
if __name__ == "__main__":
    n_bins = 6
    days = (1, 7)
    logging_root = 'exps'

    whitening = WhiteningNDVI()
    hist = HistQuantAnalysis(box_id='box3', logging_root=logging_root)
    transforms = ComposeTransform([Calibration(), ExtractRightHalf()])
    ds = dataset.Dataset('ds/dataset.json', transforms)
    for ndvi_t, ndvi_g, reg_label, clas_label, name in ds:
        ndvi_t, ndvi_g, name = whitening(ndvi_t, ndvi_g, name)
        hist(ndvi_t[0], ndvi_g[0], name)
    
    hist.plot_common_histogram()
    hist.quantize_analysis(HistQuantStatic(n_bins,    logging_root=logging_root), days)
    hist.quantize_analysis(HistQuantMultiOtsu(n_bins, logging_root=logging_root), days)
    hist.quantize_analysis(HistQuantLocalMins(n_bins, logging_root=logging_root), days)
    hist.quantize_analysis(HistQuantMRI(n_bins,       logging_root=logging_root), days)