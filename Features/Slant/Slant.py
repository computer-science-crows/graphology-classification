import cv2
import numpy as np

# Algorithm from Predicting the Big Five personality traits from handwriting, Mihai Gavrilescu and Nicolae Vizireanu

EXTREME_LEFT = -2
MODERATE_LEFT = -1
VERTICAL = 0
MODERATE_RIGHT = 1
EXTREME_RIGHT = 2


def SlantFeature(img):

    # img = cv2.imread(image_path)

    # # Convert image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define range of angles to check
    angles = np.arange(-20, 21)

    # Initialize variables
    max_sum = -1
    best_angle = 0
    
    # Iterate over angles and calculate pixel density histogram for each angle
    for angle in angles:
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        hist = np.sum(rotated, axis=0)
        
        # Normalize histogram
        hist_norm = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
        
        # Calculate sum of normalized histogram
        hist_sum = np.sum(hist_norm)
        
        # Check if this angle has the highest sum so far
        if hist_sum > max_sum:
            max_sum = hist_sum
            best_angle = angle
    
    #print(best_angle)

    # Classify slant based on best angle
    if best_angle > -2 and best_angle < 2:
        return VERTICAL
    elif best_angle >= -7 and best_angle <= -2:
        return MODERATE_RIGHT
    elif best_angle < -7 :
        return EXTREME_RIGHT
    elif best_angle >= 2 and best_angle <= 7:
        return MODERATE_LEFT
    else:
        return EXTREME_LEFT


