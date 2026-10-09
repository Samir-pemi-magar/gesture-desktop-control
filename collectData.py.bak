import cv2
import csv
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cameraTest import cameraStart

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

KEY_MAP = {
    ord('o'): 'open_palm',
    ord('f'): 'fist',
    ord('p'): 'point',
    ord('i'): 'pinch',
    ord('t'): 'four_Fingers',
    ord('v'): 'two_Fingers',
    ord('u'): 'thumb_up',
    ord('m'): 'three_Fingers'
}
counts = {label: 0 for label in KEY_MAP.values()}


def save_row(relative, label):
    row = [coord for point in relative for coord in point]
    row.append(label)
    with open('gesture_data.csv', 'a', newline='') as f:
        csv.writer(f).writerow(row)


def processFrame(frame):
    RGB_Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    MP_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_Frame)
    results = detector.detect(MP_Image)

    key = cv2.waitKey(1) & 0xFF

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            points = []
            for lm in hand:
                h, w, _ = frame.shape
                points.append((int(lm.x * w), int(lm.y * h)))

            wrist = points[0]
            mid_mcp = points[9]
            ref_dist = math.dist(wrist, mid_mcp)
            if ref_dist == 0:
                continue

            # Same normalization as your app
            relative = [
                (round((x - wrist[0]) / ref_dist, 4), round((y - wrist[1]) / ref_dist, 4))
                for x, y in points
            ]

            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
            for x, y in points:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if key in KEY_MAP:
                gesture = KEY_MAP[key]
                save_row(relative, gesture)
                counts[gesture] += 1
                print(f"Saved '{gesture}' — total: {counts[gesture]}")

    y_pos = 30
    for k, label in KEY_MAP.items():
        cv2.putText(frame, f"[{chr(k)}] {label}: {counts[label]}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        y_pos += 28

    cv2.imshow("preview", frame)


cameraStart(processFrame)