import cv2
from detection import load_detector, detect_players
from tracking import init_tracker, track_players
from distance import DistanceCalculator

detector = load_detector("models/yolov8n.pt")
tracker = init_tracker()
distance_calc = DistanceCalculator()

cap = cv2.VideoCapture("data/videos/football.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detect_players(detector, frame)
    detections = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        detections.append(([x1, y1, x2-x1, y2-y1], 0.9, "player"))

    tracks = track_players(tracker, detections, frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        player_id = track.track_id
        bbox = track.to_ltrb()
        center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

        distance_calc.update(player_id, center)
        dist = distance_calc.get_distance(player_id)

        cv2.putText(frame, f"ID {player_id}: {dist:.2f}px",
                    (int(bbox[0]), int(bbox[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])), (255,0,0), 2)

    cv2.imshow("Football Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
