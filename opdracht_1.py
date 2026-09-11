import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Verwerk een foto van de schurk')
parser.add_argument('image_path', type=str, help='Pad naar de afbeelding van de schurk')

args = parser.parse_args()
image = cv2.imread(args.image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()