from handwritten_api import HandWritten
import cv2

file = '/Users/zzmacbookpro/Desktop/BGS-data/line_segment/11447_13376792/11447_13376792-018.png'
detector = HandWritten()
text = detector.detect(img_path = file)
print('Recognized: ' + text)