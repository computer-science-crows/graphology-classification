try:
    from Baseline import Baseline
    from LineSpace import LineSpace
    from WordSpace import WordSpace
    from Margin import Margin
    from Slant import Slant
except: 
    from Features.Baseline import Baseline    
    from Features.LineSpace import LineSpace
    from Features.WordSpace import WordSpace
    from Features.Margin import Margin
    from Features.Slant import Slant
import cv2

class FeaturesInfo():
    def __init__(self, image):
        img1, img2 = FeaturesInfo.preprocessing(image)
        self.image = img1 
        self.baseline, angle = Baseline.BaselineFeature(img2)
        imgbl = Baseline.rotate(img1, angle)
        self.word_space = WordSpace.WordSpaceFeature(imgbl)
        imgbl2 = Baseline.rotate(img2, angle) 
        self.line_space = LineSpace.LineSpaceFeature(imgbl2)
        self.margin = Margin.MarginFeature(imgbl2)
        self.slant = Slant.SlantFeature(img)

    
    def thresholding(img):
        thresh, im_bw = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh, im_bw2 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return im_bw, im_bw2
    
    def preprocessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5,5),0)
        img1, img2 = FeaturesInfo.thresholding(img)
        return img1, img2

img = cv2.imread("C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/Features/Aida_2.jpg")
FeaturesInfo(img)

