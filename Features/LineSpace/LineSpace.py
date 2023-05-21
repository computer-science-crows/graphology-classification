import cv2
import numpy as np
import os
import matplotlib as plt
import scipy.signal as scy

# files_names = os.listdir("./photo-scanner/Laura")

img = cv2.imread(f"C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/Features/LineSpace/Aida.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5),0)

def thresholding(img):
        thresh, im_bw = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh, im_bw2 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
        return im_bw, im_bw2

def margin_text(img):
    img[img == 0] = 1
    img[img == 255] = 0

    proportion = int((img.shape[1])/20)
    d = [0] * (img.shape[1]//proportion +1)
    for i in range(len(d)):
        d[i] = np.sum(img[0:img.shape[0],i*proportion:min(i*proportion + proportion, img.shape[1])])/ ((min(i*proportion + proportion, img.shape[1]) - i*proportion)* img.shape[0])
             
    delta = [0] * len(d)
    th =  np.median(d)/2
    for i in range(len(d)):
        delta[i] = int((1+ np.sign(d[i] - th))//2)
 
    return delta

def SPR(img):
    PR = np.sum(img==1, axis=1)
    box = np.ones(100)
    SPR = np.convolve(PR,box, mode= 'same')
    return SPR


def gap_text_clasifier(img,delta):
    proportion = int((img.shape[1])/10)
    vertical_lines =[[None]* 2 for _ in range(len(delta))]
    
    for i in range(len(delta)):
        if delta[i]>0:
            vertical_lines[i][0] = []
            vertical_lines[i][1] = []
            
            Spr = SPR(img[0:img.shape[0],i*proportion:min(i*proportion + proportion, img.shape[1])])
            
            peaks_max = scy.find_peaks(Spr)
            peaks_min = scy.find_peaks(-Spr)
            peaks = np.concatenate((peaks_max[0],peaks_min[0]))
            peaks.sort(kind="mergesort")
            
            plt.pyplot.imshow(img[0:img.shape[0],i*proportion:min(i*proportion + proportion, img.shape[1])], cmap="binary")
            current = 0
            for j in range(len(peaks)-1):
                mp = peaks[j]+ (peaks[j+1]-peaks[j])//2
                vertical_lines[i][current].append((peaks[j], peaks[j+1], peaks[j+1]-peaks[j],mp))
                current = current==0
                if current ==0:
                    plt.pyplot.plot([0,min(i*proportion + proportion, img.shape[1])-i*proportion],[peaks[j]+ (peaks[j+1]-peaks[j])//2,peaks[j]+ (peaks[j+1]-peaks[j])//2])

            plt.pyplot.plot(Spr, range(0,img.shape[0]), color='red')
            plt.pyplot.scatter([Spr[i] for i in peaks], peaks,[20]*len(peaks))
            plt.pyplot.show()

    return vertical_lines

def mj(vertical_lines):
    m_0 = [0] * len(vertical_lines)
    m_1 = [0] * len(vertical_lines)

    for i,line in enumerate(vertical_lines):
        _,_,m_0[i],_ = np.mean(line[0]) 
        _,_,m_1[i],_ = np.mean(line[1])

    return m_0, m_1

# def spr(img, delta, PR):
#     w = [0] * 5
#     e = 0
#     for i in range(-2,3):
#         e+= np.exp((-3*abs(i))/6)
#     j=0
#     for i in range(-2,3):
#         w[j] = np.exp((-3*abs(i))/6)/e
#         j+=1


#     SPR =[0] * len(PR)
#     for i in range(len(SPR)):
#         for j in range(-2,3):
#             try:
#                 SPR[i] += d[i+j] * w[j] * PR[i+j]
#             except:
#                 pass

#     return SPR

    


thresh_img, thresh_img2 = thresholding(img)
cv2.imwrite(f"C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/Features/LineSpace/Aidax.jpg", thresh_img2)
delta = margin_text(thresh_img2)
lines = gap_text_clasifier(thresh_img2, delta)
m0, m1 = mj(lines)


