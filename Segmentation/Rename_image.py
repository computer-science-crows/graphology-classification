import cv2
import numpy as np
import os

files_names = os.listdir("./photo-cropped")

for member in files_names:
    for person in os.listdir(f"./photo-cropped/{member}"):
        i=0
        name = person[0:len(person)-4]
        for image in os.listdir(f"./photo-cropped/{member}/{person}"):
            try:
                img = cv2.imread(f"./photo-cropped/{member}/{person}/{image}")
                cv2.imwrite(f'./photo-cropped/{member}/{person}/{name}_{i}.jpg', img)
                i+=1
            except:
                print(f"./photo-cropped/{member}/{person}/{image}")

           