import cv2
import numpy as np
import os

files_names = os.listdir("./photo-scanner/Laura")


for image in files_names:

    img = cv2.imread(f"./photo-scanner/Laura/{image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try: 
        # creating a folder named data
        if not os.path.exists(f'./photo-cropped/Laura(OK)'):
            os.makedirs(f'./photo-cropped/Laura(OK)')

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')


    def thresholding(img):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh, im_bw = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            thresh, im_bw2 = cv2.threshold(img_gray, 4, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            return im_bw, im_bw2

    thresh_img, thresh_img2 = thresholding(img)
    # cv2.imwrite(f'thresh.jpg', thresh_img)
    #dilatation 
    kernel = np.ones((1,90), np.uint8)
    dilated = cv2.dilate(thresh_img,kernel,iterations =1)
    
    # kernel = np.ones((2,10), np.uint8)
    # erode = cv2.erode(dilated,kernel,iterations =1)
    # cv2.imwrite(f'dilated.jpg', dilated)


    contours, hierarchy = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours,key = lambda ctr : cv2.boundingRect(ctr)[1])

    i=0
    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img, (x,y),(x+w,y+h), (40,100,250),2)
        if x+ w> thresh_img2.shape[1] or y+h> thresh_img2.shape[0]:continue
        cv2.imwrite(f'./photo-cropped/Laura(OK)/S_{i}.jpg', thresh_img2[y:y+h,x:x+w])
        i+=1

    cv2.imwrite(f'./photo-cropped/Laura(OK)/{image}', thresh_img2)