import cv2
import numpy as np
import matplotlib.pyplot as plt


WIDE_SPACE = 0
NARROW_SPACE = 1

def WordSpaceFeature(img): #img es imagen en binario con los colores invertidos 
    
    #Nos quedamos con la parte central de la imagen
    proportion = int((img.shape[0])/4)
    img2 = img[proportion: 3*proportion, 0:img.shape[1]]

    #Aplicamos filtros mofologicos de difusion
    kernel = np.ones((2,4), np.uint8)
    img2 = cv2.dilate(img2,kernel,iterations =1)
    
    #Extraccion de los rectangulos que bordean las alabras
    contours, hierarchy = cv2.findContours(img2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours,key = lambda ctr : cv2.boundingRect(ctr)[1])

    #Determinar espacio entre palabras por los extremos de los rectangulos
    words_rectangles = []
    for ctr in sorted_contours_lines:
            x,y,w,h = cv2.boundingRect(ctr)
            # cv2.rectangle(img, (x,y),(x+w,y+h), (40,100,250),2)
            if x+ w> img2.shape[1] or y+h> img2.shape[0] or w < img2.shape[1]/30:continue
            words_rectangles.append((x,x+w))
            # plt.imshow(img, cmap="binary")

    words_rectangles.sort(key = lambda x: x[0])
    mean_space = sum([words_rectangles[i+1][0] - words_rectangles[i][1] for i in range(len(words_rectangles)-1)])/(len(words_rectangles)-1)

    if mean_space >= img.shape[1]/20: #Esta proporcionpuede cambiarse
        return WIDE_SPACE
    return NARROW_SPACE


