import glob
import math
import pickle

import cv2
import numpy as np
import scipy as sp
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy import signal, special


class sleeqQA():
    '''
    sleeq,a new fully blind video  quality assessment method 
    Paper:A No-Reference Video Quality Predictor For Compression And Scaling Artifacts
    '''

    def __init__(self, patch_size=72, n_threshold=0.2, ksize=7, Bsigma=1):
        self.patch_size = patch_size  # patch size
        self.n_threshold = n_threshold  # percentile threshold
        self.ksize = ksize  # gaussblur  kernel size
        self.Bsigma = Bsigma  # kernel standard deviation

    def sleeq_video(self, filename):
        '''
        evaluate score of a video 
        '''
        cap = cv2.VideoCapture(filename)
        scores = []
        weights = []
        first = True
        while 1:
            ret, frame = cap.read()
            if frame is None:
                break
            # consider  luminance component of every frame
            nextY = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
            nextY = nextY.astype(np.float32)
            if first:
                first = False
                Y = nextY
                continue
            Ydiff = nextY-Y  # ?????
            frame_scores, frame_weights = self.sleeq(Y, Ydiff)
            scores.extend(frame_scores)
            weights.extend(frame_weights)
            Y = nextY

        score = self.spatial_temporal_pooling(scores, weights)

        return round(score, 3)

    def sleeq(self, frame, framediff):
        h, w = frame.shape
        psize = self.patch_size
        N = w//psize
        M = h//psize

        scores = []
        weights = []

        for x in range(N):
            for y in range(M):
                patch = frame[y*psize:(y+1)*psize,
                              x*psize:(x+1)*psize]
                patchdiff = framediff[y*psize:(y+1) *
                                      psize, x*psize:(x+1)*psize]
                patchQ, weight = self.__get_Q_weight(patch, patchdiff)

                scores.append(patchQ)
                weights.append(weight)

        return scores, weights

    def spatial_temporal_pooling(self, scores, weights):
        res = list(zip(weights, scores))
        res = sorted(res)
        scores_order = list(zip(*res))[1]

        return np.mean(scores_order[int(self.n_threshold*len(scores_order)):])

    def __get_alpha_sigma(self, patch):
        patch_blur = cv2.GaussianBlur(
            patch, (self.ksize, self.ksize), self.Bsigma)

        mscn, var = self.__calculate_mscn_coefficients(patch)
        alpha = self.__ggd_features(mscn)

        mscn_blur, var_blur = self.__calculate_mscn_coefficients(patch_blur)
        alpha_blur = self.__ggd_features(mscn_blur)

        # return score and weight
        return abs(alpha-alpha_blur), abs(var-var_blur)

    def __get_Q_weight(self, patch, patchdiff):
        # mp = np.abs(np.mean(patchdiff/255.0))
        mp = np.mean(np.abs(patchdiff)/255.0)  # ??
        alpha_s, delta_var = self.__get_alpha_sigma(patch)
        if mp < 0.001:
            return alpha_s, delta_var
        alpha_t, _ = self.__get_alpha_sigma(patchdiff)
        Q = (1-mp)*alpha_s+mp*alpha_t

        return Q, delta_var

    def __ggd_features(self, mscn):
        '''
            Paper:Estimation of Shape Parameter for Generalized Gaussian Distributions in Subband Decompositions of Video
        '''
        gamma_range = np.arange(0.2, 10, 0.001)
        a = special.gamma(2.0/gamma_range)
        a *= a
        b = special.gamma(1.0/gamma_range)
        c = special.gamma(3.0/gamma_range)
        prec_gammas = a/(b*c)

        nr_gam = 1/prec_gammas
        sigma = np.var(mscn)
        E = np.mean(np.abs(mscn))
        # if(E == 0):
        #     import pdb
        #     pdb.set_trace()
        rho = sigma/E**2
        pos = np.argmin(np.abs(nr_gam - rho))

        return gamma_range[pos]  # GGD model alpha parameter to decide shape

    def __calculate_mscn_coefficients(self, dis_image):
        dis_image = dis_image.astype(np.float32)
        ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
        ux_sq = ux*ux
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(
            dis_image**2, (7, 7), 7/6)-ux_sq))
        # sigma = np.sqrt(np.abs(cv2.GaussianBlur(
        #     (dis_image-ux)**2, (7, 7), 7/6)))
        mscn = (dis_image-ux)/(1+sigma)

        return mscn, np.mean(sigma)


'''
    def __calculate_mscn_coefficients2(self, image, kernel_size=6, sigma=7/6):
        C = 1
        kernel = self.__gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = self.__signal.convolve2d(image, kernel, 'same')
        local_var = self.__local_deviation(image, local_mean, kernel)

        return (image - local_mean) / (local_var + C), np.mean(local_var)

    def __normalize_kernel(self, kernel):
        return kernel / np.sum(kernel)

    def __gaussian_kernel2d(self, n, sigma):
        Y, X = np.indices((n, n)) - int(n/2)
        gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * \
            np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

        return self.__normalize_kernel(gaussian_kernel)

    def __local_mean(self, image, kernel):
        return signal.convolve2d(image, 8, 'same')

    def __local_deviation(self, image, local_mean, kernel):
        "Vectorized approximation of local deviation"
        sigma = image ** 2
        sigma = signal.convolve2d(sigma, kernel, 'same')

        return np.sqrt(np.abs(local_mean ** 2 - sigma))
'''
