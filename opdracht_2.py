import argparse

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description="Maak een ansichtkaart voor de schurk")
parser.add_argument("background_path", type=str, help="Pad naar de achtergrondafbeelding")
args = parser.parse_args()

image = Image.open(args.background_path)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 50)
draw.text((300, 370), "Beterschap! Hou vol daar binnen.", fill="black", font=font)

image.save("kaartje.png")
print("Ansichtkaart opgeslagen als kaartje.png")