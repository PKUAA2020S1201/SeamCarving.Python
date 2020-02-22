# encoding = utf-8

import tqdm
import numba
import warnings
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve


warnings.simplefilter('ignore', numba.NumbaWarning)


class SeamCarving(object):
    def __init__(self, img, shape=None, scale=None):
        assert shape or scale
        if img.dtype == np.uint8:
            self.type = np.uint8
            self.src = img.astype(np.float32) / 255
            self.dst = img.astype(np.float32) / 255
        else:
            self.type = np.float32
            self.src = img.copy()
            self.dst = img.copy()
        if shape:
            self.shape = list(img.shape)
            for idx, val in enumerate(shape):
                if val and val != -1:
                    self.shape[idx] = val
        else:
            self.shape = list(img.shape)
            if type(scale) in [list, tuple]:
                for idx, val in enumerate(scale):
                    if val and val != -1:
                        self.shape[idx] = int(self.shape[idx] * val)
            else:
                self.shape[0] = int(self.shape[0] * scale)
                self.shape[1] = int(self.shape[1] * scale)
    
    def run(self):
        step = lambda img: self.remove_seam_v(img, self.seam_mask_v(self.energy_map(img)))
        for i in tqdm.tqdm(range(img.shape[1] - self.shape[1])):
            self.dst = step(self.dst)
        step = lambda img: self.remove_seam_h(img, self.seam_mask_h(self.energy_map(img)))
        for i in tqdm.tqdm(range(img.shape[0] - self.shape[0])):
            self.dst = step(self.dst)
    
    def show(self):
        def plt_imshow(img):
            plt.figure()
            plt.imshow((img * 255).astype(np.uint8))
        plt_imshow(self.src)
        plt_imshow(self.dst)
        plt.show()

    def result(self):
        if self.type == np.uint8:
            return (self.dst * 255).astype(np.uint8)
        else:
            return self.dst.copy()

    @staticmethod
    @numba.jit
    def energy_map(img: np.ndarray) -> np.ndarray:
        kernel_v = np.stack([np.array([
            [ 1,  2,  1], 
            [ 0,  0,  0], 
            [-1, -2, -1]
        ], dtype=np.float32)] * img.shape[2], axis=2)
        kernel_h = np.stack([np.array([
            [ 1,  0, -1], 
            [ 2,  0, -2], 
            [ 1,  0, -1]
        ], dtype=np.float32)] * img.shape[2], axis=2)
        conv_v = convolve(img, kernel_v)
        conv_h = convolve(img, kernel_h)
        energy_map = np.sum(np.square(conv_v) + np.square(conv_h), axis=2)
        return energy_map
    
    @staticmethod
    @numba.jit
    def seam_mask_v(eng: np.ndarray) -> np.ndarray:
        n, m = eng.shape
        p = np.zeros_like(eng, dtype=np.int)
        f = np.zeros_like(eng, dtype=np.float32)
        seam_mask = np.ones_like(eng, dtype=np.bool)
        for i in range(1, n):
            for j in range(0, m):
                l = max(0, j - 1)
                r = min(m, j + 2)
                k = np.argmin(f[i - 1, l : r]) + l
                f[i, j] = f[i - 1, k] + eng[i, j]
                p[i, j] = k
        j = np.argmin(f[-1])
        for i in range(n - 1, -1, -1):
            seam_mask[i, j] = 0
            j = p[i, j]
        return seam_mask
    
    @staticmethod
    @numba.jit
    def seam_mask_h(eng: np.ndarray) -> np.ndarray:
        n, m = eng.shape
        p = np.zeros_like(eng, dtype=np.int)
        f = np.zeros_like(eng, dtype=np.float32)
        seam_mask = np.ones_like(eng, dtype=np.bool)
        for j in range(1, m):
            for i in range(1, n):
                l = max(0, i - 1)
                r = min(n, i + 2)
                k = np.argmin(f[l : r, j - 1]) + l
                f[i, j] = f[k, j - 1] + eng[i, j]
                p[i, j] = k
        i = np.argmin(f[:, -1])
        for j in range(m - 1, -1, -1):
            seam_mask[i, j] = 0
            i = p[i, j]
        return seam_mask

    @staticmethod
    @numba.jit
    def remove_seam_v(img, msk):
        n, m, c = img.shape
        msk = np.stack([msk] * c, axis=2)
        img = img[msk].reshape((n, m - 1, c))
        return img

    @staticmethod
    @numba.jit
    def remove_seam_h(img, msk):
        n, m, c = img.shape
        msk = np.stack([msk] * c, axis=2)
        img = img[msk].reshape((n - 1, m, c))
        return img


if __name__ == '__main__':
    img = imread('images/boat.jpg')
    sc = SeamCarving(img, scale=(0.8, 0.8))
    sc.run()
    # sc.show()
    res = sc.result()
    imwrite('images/boat_result.jpg', res)
