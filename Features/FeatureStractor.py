from .Baseline import Baseline
from .LineSpace import LineSpace
from .WordSpace import WordSpace
from .Margin import Margin

class FeaturesInfo():
    def __init__(self, image):
        img1, img2 = FeaturesInfo.preprocessing(image)
        self.image = img1 
        self.baseline = Baseline.BaselineFeature(img2)
        self.word_space = WordSpace.WordSpaceFeature(img1) 
        self.line_space = LineSpace.LineSpaceFeature(img2)
        self.margin = Margin.MarginFeature(img2)

    
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