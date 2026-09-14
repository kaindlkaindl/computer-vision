import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--image", type = str)
args = parser.parse_args()

img = cv2.imread(args.image)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.rectangle(gray_img, (160, 200), (420, 260), (0, 0, 0), -1)

cv2.imwrite("schurk-anoniem.png", gray_img)
