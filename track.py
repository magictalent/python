import cv2
import numpy as np
from ultralytics import YOLO
from trackers.deep_sort import DeepSort
from utils.distance_utils import calculate_distance

# Load YOLOv8
yolo_model = YOLO("models/yolov8n.pt")

# Load DeepSORT
deepsort = DeepSort("models/deepsort/mars-small128.pb")

# Open video
cap = cv2.VideoCapture("data/football.mp4")

# Store paths for each player
player_paths = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect players
    results = yolo_model(frame)[0]

    detections = []
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, 1.0])

    # Track players
    tracks = deepsort.update_tracks(detections, frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Save path
        if track_id not in player_paths:
            player_paths[track_id] = []
        player_paths[track_id].append((cx, cy))

        # Draw box + ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw path
        for i in range(1, len(player_paths[track_id])):
            cv2.line(frame, player_paths[track_id][i - 1], player_paths[track_id][i], (255, 0, 0), 2)

    cv2.imshow("Football Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Show distances
for pid, path in player_paths.items():
    dist = calculate_distance(path)
    print(f"Player {pid} ran approx: {dist:.2f} meters")
