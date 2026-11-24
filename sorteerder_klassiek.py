import math
import random
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

PROJECT_DIR = Path.cwd()
DATASET_DIR = PROJECT_DIR / "dataset"

IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
YAML_PATH = DATASET_DIR / "data.yaml"

RESULTS_DIR = PROJECT_DIR / "resultaten"

print("Dataset:", DATASET_DIR)
print("Afbeeldingen:", IMAGES_DIR.exists())
print("Labels:", LABELS_DIR.exists())
print("YAML:", YAML_PATH.exists())
print("Aantal foto's:", len([
    p for p in IMAGES_DIR.rglob("*")
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
]) if IMAGES_DIR.exists() else 0)

def load_classes():
    with YAML_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]

    if isinstance(names, dict):
        names = [
            name for _, name in sorted(
                names.items(),
                key=lambda item: int(item[0])
            )
        ]

    return tuple(names)


CLASSES = (
    'hex_nut',
    'washer',
    'allen_screw',
    'hex_bolt',
    'carriage_bolt',
    'barrel_nut',
    'pneumatic_fitting'
)

def choose_class(prefixes, size_mm=None):
    vertaling = {
        "zeskantmoer_": "hex_nut",
        "carrosseriering_": "washer",
        "sluitring_": "washer",
        "inbusbout_": "allen_screw",
        "tapbout_": "hex_bolt",
        "zeskantbout_": "hex_bolt",
        "slotbus_": "carriage_bolt",
        "montagebus_": "barrel_nut",
        "fitting_": "pneumatic_fitting",
    }

    for prefix in prefixes:
        if prefix in vertaling:
            klasse = vertaling[prefix]
            if klasse in CLASSES:
                return klasse

    return "onbekend"


def separate_touching_objects(binary_mask, min_area=300):
    """
    Separate touching parts using Distance Transform and Watershed segmentation.
    Computes distance to background, finds distinct local cores, and segments
    merged objects when multiple prominent peaks exist.
    """
    # Remove small speckle noise with light morphological opening
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    raw_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    separated_contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        blob_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1, offset=(-x, -y))

        # Distance Transform: calculates Euclidean distance to closest background pixel
        dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist <= 0:
            continue

        # Threshold cores at 48% of peak distance to detect separated centers
        _, cores = cv2.threshold(dist, 0.48 * max_dist, 255, cv2.THRESH_BINARY)
        cores = np.uint8(cores)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cores)
        valid_labels = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 14]
        core_areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_labels]

        # Watershed split when exactly two distinct substantial cores exist
        should_split = (
            len(valid_labels) == 2
            and area >= 5000
            and max(core_areas) < area * 0.90
            and min(core_areas) >= 40
        )

        if should_split:
            markers = np.zeros((bh, bw), dtype=np.int32)
            for idx, lbl in enumerate(valid_labels):
                markers[labels == lbl] = idx + 2
            markers[blob_mask == 0] = 1  # 1 is background

            dummy_bgr = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
            cv2.watershed(dummy_bgr, markers)

            for idx, _ in enumerate(valid_labels):
                part_mask = np.uint8(markers == (idx + 2))
                part_cnts, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if part_cnts:
                    shifted = part_cnts[0] + np.array([x, y])
                    if cv2.contourArea(shifted) >= min_area:
                        separated_contours.append(shifted)
        else:
            separated_contours.append(cnt)

    return separated_contours


def detect_parts(image_path, pixels_per_mm=8.6):
    """
    Process an image and classify all detected mechanical parts using classical CV.
    
    Returns:
        tuple: (pandas.DataFrame of detected parts, annotated visualization image)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return None, None

    # Resize to standard width of 1600 px for uniform scale and fast processing
    h_orig, w_orig = img.shape[:2]
    scale = 1600.0 / w_orig
    small_img = cv2.resize(img, (1600, int(h_orig * scale)))
    vis_img = small_img.copy()

    # ── Illumination-robust Background Segmentation (LAB Color Space) ──────────
    # Estimate global background color from image margins
    lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    border_samples = np.concatenate((
        lab[20:70, 70:w-70].reshape(-1, 3),
        lab[h-70:h-20, 70:w-70].reshape(-1, 3)
    ), axis=0)
    bg_color = np.median(border_samples, axis=0)

    # Heavily blurred image approximates slow illumination / vignetting gradients
    smooth_bg = cv2.GaussianBlur(lab, (0, 0), 95)
    local_diff = np.linalg.norm(lab - smooth_bg, axis=2)
    color_diff = np.linalg.norm((lab - bg_color)[..., 1:3], axis=2)
    darkness = np.maximum(0, smooth_bg[..., 0] - lab[..., 0])

    # Combined foreground score
    score = np.maximum.reduce((local_diff * 0.85, color_diff * 1.8, darkness * 1.2))
    score = cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)
    binary = np.uint8(score > 24) * 255

    # Light morphological close fills small holes without merging neighboring bolts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debug mask output
    cv2.imwrite(str(PROJECT_DIR / "debug_mask.png"), binary)

    img_name = Path(image_path).name

    detected_parts = []
    visualization_items = []
    object_number = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > (small_img.shape[0] * small_img.shape[1] * 0.35):
            continue

        # Centroid from spatial moments
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Frame border filter (skip partial contours cut off at boundaries)
        if cx < 25 or cy < 25 or cx > small_img.shape[1] - 25 or cy > small_img.shape[0] - 25:
            continue

        # Rotated bounding box
        rect = cv2.minAreaRect(cnt)
        _, (box_w, box_h), angle = rect
        length = max(box_w, box_h)
        width = min(box_w, box_h)
        if width == 0:
            continue

        # Standardize angle to align with the major (longer) axis
        if box_w < box_h:
            angle += 90.0
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        aspect_ratio = length / width
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        length_mm = (length / scale) / pixels_per_mm
        width_mm = (width / scale) / pixels_per_mm

        # Check for central hole (distinguishes washers/nuts from solid cylindrical pins)
        x_min, y_min, bw, bh = cv2.boundingRect(cnt)
        contour_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1, offset=(-x_min, -y_min))
        local_fg = binary[y_min:y_min + bh, x_min:x_min + bw]
        mask_pixels = int(np.count_nonzero(contour_mask > 0))
        hole_pixels = int(np.count_nonzero((contour_mask > 0) & (local_fg == 0)))
        has_hole = (hole_pixels / mask_pixels if mask_pixels else 0.0) > 0.035

        # In-contour color check: detect vivid blue push-to-connect ring on pneumatic fittings
        hsv_full = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_full, (90, 60, 50), (135, 255, 255))
        obj_mask = np.zeros(small_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
        blue_in_obj = cv2.bitwise_and(blue_mask, blue_mask, mask=obj_mask)
        obj_pixels = cv2.countNonZero(obj_mask)
        blue_ratio = cv2.countNonZero(blue_in_obj) / max(1, obj_pixels)
        is_fitting = blue_ratio > 0.03

        # Detect merged clusters (large non-elongated blobs)
        is_cluster = area > 15000 and aspect_ratio < 2.0 and circularity < 0.45

        # ── Classification into the 11 classes ────────────────────────────────
        if is_cluster:
            part_class = "uncertain_cluster"
            color = (0, 100, 255)       # Red-orange
        elif is_fitting:
            part_class = choose_class(("fitting_",), length_mm)
            color = (255, 0, 200)       # Magenta
        elif aspect_ratio < 1.45 and (has_hole or circularity >= 0.76):
            part_class = choose_class(("carrosseriering_",), max(length_mm, width_mm))
            color = (0, 230, 255)       # Yellow
        elif aspect_ratio < 1.35:
            part_class = choose_class(("zeskantmoer_",), max(length_mm, width_mm))
            color = (255, 130, 0)       # Blue-orange
        else:
            part_class = choose_class(
                ("inbusbout_", "montagebus_", "slotbus_", "tapbout_"),
                length_mm,
            )
            color = (0, 220, 220)       # Cyan

        # Convert to physical millimeter coordinates (for robotic pick-and-place)
        pos_x_mm = round((cx / scale) / pixels_per_mm, 1)
        pos_y_mm = round((cy / scale) / pixels_per_mm, 1)
        length_mm = round(length_mm, 1)
        width_mm = round(width_mm, 1)

        # Draw oriented bounding box
        box_pts = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(vis_img, [box_pts], 0, color, 2, cv2.LINE_AA)

        # Draw circular ID badge at centroid
        cv2.circle(vis_img, (cx, cy), 17, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(vis_img, (cx, cy), 15, color, -1, cv2.LINE_AA)
        id_text = str(object_number)
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(
            vis_img, id_text,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw orientation arrow along the major axis
        angle_rad = math.radians(angle)
        arrow_len = max(20, int(length * 0.25))
        dx = int(arrow_len * math.cos(angle_rad))
        dy = int(arrow_len * math.sin(angle_rad))
        cv2.arrowedLine(
            vis_img, (cx, cy), (cx + dx, cy + dy), (0, 0, 255),
            2, cv2.LINE_AA, tipLength=0.35
        )

        detected_parts.append({
            "image": img_name,
            "object_id": object_number,
            "class": part_class,
            "position_x_px": int(cx / scale),
            "position_y_px": int(cy / scale),
            "position_x_mm": pos_x_mm,
            "position_y_mm": pos_y_mm,
            "angle_degrees": round(angle, 1),
            "length_mm": length_mm,
            "width_mm": width_mm,
            "aspect_ratio": round(aspect_ratio, 2),
            "circularity": round(circularity, 2),
            "method": "Classical_OpenCV"
        })

        visualization_items.append({
            "object_id": object_number,
            "class": part_class,
            "color": color,
            "position_x_mm": pos_x_mm,
            "position_y_mm": pos_y_mm,
            "angle_degrees": angle,
        })
        object_number += 1

    df = pd.DataFrame(detected_parts)

    # ── Render Side Info Panel ─────────────────────────────────────────────────
    panel_width = 480
    header_height = 78
    row_height = 48
    panel_height = max(vis_img.shape[0], header_height + len(visualization_items) * row_height + 12)
    visualization = np.full((panel_height, vis_img.shape[1] + panel_width, 3), (28, 28, 28), dtype=np.uint8)
    visualization[:vis_img.shape[0], :vis_img.shape[1]] = vis_img

    cv2.putText(
        visualization, f"Parts: {len(visualization_items)}",
        (vis_img.shape[1] + 20, 42), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    for index, item in enumerate(visualization_items):
        y = header_height + index * row_height
        cv2.circle(visualization, (vis_img.shape[1] + 31, y + 12), 13, item["color"], -1, cv2.LINE_AA)
        cv2.putText(
            visualization, str(item["object_id"]),
            (vis_img.shape[1] + 25, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            visualization, item["class"],
            (vis_img.shape[1] + 55, y + 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (255, 255, 255), 1, cv2.LINE_AA
        )
        coords = f"x={item['position_x_mm']:.1f} mm, y={item['position_y_mm']:.1f} mm, angle={item['angle_degrees']:.0f} deg"
        cv2.putText(
            visualization, coords,
            (vis_img.shape[1] + 55, y + 29), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (190, 190, 190), 1, cv2.LINE_AA
        )

    return df, visualization


def analyze_random_image(image_path=None, show=True):

    # Zoek alle afbeeldingen in jouw dataset
    all_images = [
        p for p in IMAGES_DIR.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if image_path is None:
        if not all_images:
            raise FileNotFoundError(
                f"Geen afbeeldingen gevonden in {IMAGES_DIR}"
            )
        chosen = random.choice(all_images)
    else:
        chosen = Path(image_path)

        if not chosen.is_absolute() and not chosen.is_file():
            chosen = IMAGES_DIR / chosen

    if not chosen.is_file():
        raise FileNotFoundError(f"Afbeelding niet gevonden: {chosen}")

    print(f"Geselecteerde afbeelding: {chosen.name}")

    # Gebruik de oorspronkelijke detectiefunctie
    df, visualization = detect_parts(chosen)

    if df is None or visualization is None:
        raise RuntimeError("Afbeelding kon niet worden verwerkt.")

    if not df.empty:
        print(f"\nGedetecteerde onderdelen: {len(df)}")
        print(df.to_string(index=False))
    else:
        print("Geen onderdelen gedetecteerd.")

    # Resultaten opslaan
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vis_dir = RESULTS_DIR / "visualisaties"
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "detecties_klassiek.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)

        if "image" in existing.columns:
            existing = existing[existing["image"] != chosen.name]

        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)

    vis_path = vis_dir / f"{chosen.stem}_visual_classical.jpg"
    cv2.imwrite(str(vis_path), visualization)

    if show:
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        plt.title(f"Detecties: {chosen.name}")
        plt.axis("off")
        plt.show()

    return df, visualization

# Backward compatibility aliases
laad_klassen = load_classes
kies_klasse = choose_class
scheid_aanliggende_objecten = separate_touching_objects
detecteer_onderdelen = detect_parts
analyseer_willekeurige_foto = analyze_random_image
KLASSEN = CLASSES


df, afbeelding = analyze_random_image()