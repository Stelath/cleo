"""Generate placeholder demo images for the navigator demo.

Creates two simple scenes so the demo runs without real photos.
Replace with real images for a better demo.
"""

import cv2
import numpy as np


def draw_sidewalk_scene(path: str) -> None:
    """Draw a simple sidewalk scene with obstacles."""
    img = np.full((480, 640, 3), (180, 200, 180), dtype=np.uint8)  # light gray-green bg

    # Sky
    img[:200, :] = (200, 160, 130)  # blue-ish sky (BGR)

    # Sidewalk (gray trapezoid)
    pts = np.array([[200, 480], [440, 480], [380, 200], [260, 200]], np.int32)
    cv2.fillPoly(img, [pts], (160, 160, 160))

    # Obstacle — red box on the sidewalk
    cv2.rectangle(img, (280, 300), (360, 380), (0, 0, 200), -1)
    cv2.putText(img, "BOX", (290, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Curb edge line
    cv2.line(img, (200, 480), (260, 200), (100, 100, 100), 3)
    cv2.line(img, (440, 480), (380, 200), (100, 100, 100), 3)

    # Person silhouette (circle + rectangle) at 2 o'clock
    cv2.circle(img, (450, 250), 20, (50, 50, 50), -1)
    cv2.rectangle(img, (435, 270), (465, 350), (50, 50, 50), -1)

    cv2.imwrite(path, img)
    print(f"  wrote {path}")


def draw_intersection_scene(path: str) -> None:
    """Draw a simple street intersection scene."""
    img = np.full((480, 640, 3), (160, 160, 160), dtype=np.uint8)  # gray road

    # Sky
    img[:180, :] = (210, 170, 140)

    # Crosswalk stripes
    for y in range(300, 460, 30):
        cv2.rectangle(img, (220, y), (420, y + 12), (255, 255, 255), -1)

    # Traffic light post
    cv2.rectangle(img, (500, 100), (515, 300), (60, 60, 60), -1)
    # Red light
    cv2.circle(img, (508, 130), 12, (0, 0, 255), -1)
    # Green light (off)
    cv2.circle(img, (508, 170), 12, (0, 80, 0), -1)

    # Car shape (blue rectangle)
    cv2.rectangle(img, (50, 250), (180, 320), (180, 80, 30), -1)
    cv2.putText(img, "CAR", (80, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stairs ahead
    for y in range(200, 260, 15):
        cv2.rectangle(img, (250, y), (390, y + 12), (120, 120, 130), -1)
        cv2.line(img, (250, y), (390, y), (80, 80, 80), 1)

    cv2.imwrite(path, img)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Generating demo images...", flush=True)
    draw_sidewalk_scene("scripts/demo_images/sidewalk.jpg")
    draw_intersection_scene("scripts/demo_images/intersection.jpg")
    print("Done.", flush=True)
