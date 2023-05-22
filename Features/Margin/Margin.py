import cv2
import numpy as np
import matplotlib.pyplot as plt

SMALL_MARGIN = 0
BIG_MARGIN = 1 

def MarginFeature(img): #Recibe imagen en binario preprocesada 
    img[img == 0] = 0
    img[img == 255] = 1

    PR = np.sum(img, axis=0)
    percentage_part = (img.shape[0]*2)//25 #8% del ancho de la imagen

    margin_px_line = 0  
    for x in PR:
        if x > percentage_part: break
        margin_px_line +=1
    
    # plt.imshow(img, cmap="binary")
    # plt.plot([margin_px_line,margin_px_line], [0,img.shape[0]], color='red')
    # plt.show()
    if margin_px_line > img.shape[1]//25: #Este porcentaje esta sujeto a cambios
        return BIG_MARGIN
    return SMALL_MARGIN




