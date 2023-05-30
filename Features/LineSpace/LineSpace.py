import cv2
import numpy as np
import os
import matplotlib as plt
import scipy.signal as scy
import math

LINESPACE_SEPARETED = 0
LINESPACE_CROWDED = 1


def margin_text(img):
    # img[img==1] = 255
    # img[img == 0] = 1
    # img[img == 255] = 0

    proportion = int((img.shape[1])/20)
    d = [0] * (img.shape[1]//proportion + 1)
    for i in range(len(d)):
        val = np.sum(img[0:img.shape[0], i*proportion:min(i*proportion + proportion, img.shape[1])]
                     ) / ((min(i*proportion + proportion, img.shape[1]) - i*proportion) * img.shape[0])
        if not math.isnan(val):
            d[i] = val

    delta = [0] * len(d)
    th = np.median(d)/2
    for i in range(len(d)):
        delta[i] = int((1 + np.sign(d[i] - th))//2)

    return delta, proportion


def SPR(img):

    PR = np.sum(img == 1, axis=1)
    box = np.ones((img.shape[0])//3)
    SPR = np.convolve(PR, box, mode='same')
    return SPR


def LineSpaceFeature(img):

    delta, proportion = margin_text(img)
    width = 0
    total = 0

    for i in range(len(delta)):
        if delta[i] > 0:

            sub_img = img[0:img.shape[0], i *
                          proportion:min(i*proportion + proportion, img.shape[1])]
            Spr = list(SPR(sub_img))

            peaks_max, _ = scy.find_peaks(Spr, distance=img.shape[1]//10)
            peaks_min, _ = scy.find_peaks(
                [Spr[i] * -1 for i in range(len(Spr))], distance=img.shape[1]//10)
            peaks = np.concatenate((peaks_max, peaks_min))
            peaks.sort(kind="mergesort")

            # plt.pyplot.imshow(sub_img, cmap="binary")

            # TODO en dependencia de la cantidad de pikes si 1 espaciado amplo, si 2 calcular densidad de extremos, si >2 descartar
            # TODO asignrle a la imagen el promedio

            if len(peaks) == 1:
                width += 1
                total += 1
            elif len(peaks) == 2:
                width += np.sum(sub_img[0:peaks[0], 0:sub_img.shape[1]]) + np.sum(
                    sub_img[0:peaks[1], 0:sub_img.shape[1]]) <= (sub_img.shape[0] * sub_img.shape[1])/5
                total += 1

            # plt.pyplot.plot(Spr[0:len(Spr)], range(0,sub_img.shape[0]), color='red')
            # plt.pyplot.scatter([Spr[i] for i in peaks], peaks,[20]*len(peaks))
            # plt.pyplot.show()

    if width/(total) < 0.7:
        return LINESPACE_CROWDED
    return LINESPACE_SEPARETED
