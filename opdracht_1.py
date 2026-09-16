import argparse
import cv2
import numpy as np
import math

def anonymize(img, eye1, eye2):
    distance = (math.dist(eye1, eye2))
    thickness = int(distance * 0.35)
    result = img.copy()
    cv2.line(result, eye1, eye2, (0, 0, 0), thickness)
    return result

parser = argparse.ArgumentParser(description='Verwerk een foto van de schurk')
parser.add_argument('image_path', type=str, help='Pad naar de afbeelding van de schurk')

args = parser.parse_args()
image = cv2.imread(args.image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eye1 = (200, 250)
eye2 = (360, 230)

result = anonymize(gray_image, eye1, eye2)

cv2.imshow('Grayscale', result)
cv2.waitKey(0)  
cv2.destroyAllWindows()
cv2.imwrite('verwerkte_schurk.jpg', result)
