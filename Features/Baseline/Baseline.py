import cv2
import numpy as np
import matplotlib as plt

ASCENDING = 1
LEVELED = 0
DESCENDING =-1

def rotate(img, angle):
    M  = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
    return cv2.warpAffine(img, M, (img.shape[1],img.shape[0]),borderValue = 0)


def BaselineFeature(img):
    
    img[img == 0] = 1
    img[img == 255] = 0

    center = img.shape[0]//2 
    top = center - img.shape[0]//5
    bottom = center + img.shape[0]//5

    best = 0 
    angle = 0
    for i in range(-30,30):
        imgx = rotate(img, i)
        s = np.sum(imgx[top:bottom, 0:img.shape[1]])
        if s > best:
            best = s
            angle = i
            plt.pyplot.imshow(imgx, cmap="binary")
    
    if angle < -5:
        return ASCENDING, angle
    elif angle > 5:
        return DESCENDING, angle
    return LEVELED, angle

