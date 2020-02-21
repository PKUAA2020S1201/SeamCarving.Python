# encoding = utf-8

import tqdm
import numba
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve


class SeamCarving(object):
    def __init__(self, img, shape=None, scale=None):
        assert shape or scale
        assert img.dtype == np.uint8 and len(img.shape) == 3
        self.src = img.astype(np.float32) / 255
        self.dst = img.astype(np.float32) / 255
        if shape:
            self.shape = shape
        else:
            self.shape = img.shape
            self.shape[0] = int(self.shape[0] * scale)
            self.shape[1] = int(self.shape[1] * scale)
    
    def run(self):
        step = lambda img: self.remove_seam(img, self.seam_mask(self.energy_map(img)))
        for i in tqdm.tqdm(range(img.shape[1] - self.shape[1])):
            self.dst = step(self.dst)
        self.dst = self.dst.T
        for i in tqdm.tqdm(range(img.shape[0] - self.shape[0])):
            self.dst = step(self.dst)
        self.dst = self.dst.T
    
    def show(self):
        def plt_imshow(img):
            plt.figure()
            plt.imshow((img * 255).astype(np.uint8))
        plt_imshow(self.src)
        plt_imshow(self.dst)
        plt.show()

    @staticmethod
    @numba.jit
    def energy_map(img):
        assert img.dtype == np.float32 and len(img.shape) == 3
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
    def seam_mask(eng):
        assert eng.dtype == np.float32 and len(eng.shape) == 2
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
        for i in reversed(range(0, n)):
            seam_mask[i, j] = 0
            j = p[i, j]
        return seam_mask
    

    @staticmethod
    @numba.jit
    def remove_seam(img, msk):
        assert len(img.shape) == 3 and len(msk.shape) == 2
        n, m, c = img.shape
        msk = np.stack([msk] * c, axis=2)
        img = img[msk].reshape((n, m - 1, c))
        return img


if __name__ == '__main__':
    img = imread('images/boat.jpg')
    sc = SeamCarving(img, shape=(533, 400, 3))
    sc.run()
    sc.show()
