import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--image", type = str)
args = parser.parse_args()

img = cv2.imread(args.image)

cv2.line(img, (1000, 720), (1000, 560), (0, 255, 0), 4)
cv2.circle(img, (1021, 539), 20, (0, 245, 245), -1)
cv2.circle(img, (1021, 581), 20, (0, 245, 245), -1)
cv2.circle(img, (979, 581), 20, (0, 245, 245), -1)
cv2.circle(img, (979, 539), 20, (0, 245, 245), -1)
cv2.circle(img, (1000, 530), 20, (0, 255, 255), -1)
cv2.circle(img, (1030, 560), 20, (0, 255, 255), -1)
cv2.circle(img, (1000, 590), 20, (0, 255, 255), -1)
cv2.circle(img, (970, 560), 20, (0, 255, 255), -1)
cv2.circle(img, (1000, 560), 30, (255, 255, 255), -1)

cv2.putText(
    img,
    "Veel sterkte in de bajes\n\nGroetjes,\nGabriël en Sjoerd",
    (120, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (0, 0, 0),
    1,
)

cv2.imwrite("kaartje.png", img)
