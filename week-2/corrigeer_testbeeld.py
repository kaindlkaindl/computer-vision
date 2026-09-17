import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("image", type = str)
args = parser.parse_args()

img = cv2.imread(args.image)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.equalizeHist(v)
img_vequalized = cv2.merge([h, s, v])
img_vequalized = cv2.cvtColor(img_vequalized, cv2.COLOR_HSV2BGR)

cv2.putText(
    img_vequalized,
    "Methode:\ncv2.cvtColor(BGR2HSV)\ncv2.split(HSV)\ncv2.equalizeHist(V)\ncv2.merge([h, s, v])\ncv2.cvtColor(HSV2BGR)",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2
)

cv2.imwrite("testbeeld_vequalized.jpg", img_vequalized)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.createCLAHE(clipLimit=5.0).apply(v)
img_clahe = cv2.merge([h, s, v])
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_HSV2BGR)

cv2.putText(
    img_clahe,
    "Methode:\ncv2.cvtColor(BGR2HSV)\ncv2.split(HSV)\ncv2.createCLAHE(clipLimit=5.0).apply(v)\ncv2.merge([h, s, v])\ncv2.cvtColor(HSV2BGR)",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    2
)

cv2.imwrite("testbeeld_clahe.jpg", img_clahe)