# src/visualizer.py
import cv2

class Visualizer:
    def __init__(self, font_scale=0.5, thickness=2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_player(self, frame, track_id, bbox, distance_m):
        x1,y1,x2,y2 = map(int, bbox)
        # rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), (200, 50, 50), 2)
        # label
        label = f"ID:{track_id} {distance_m:.1f}m"
        (w, h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 6, y1), (200,50,50), -1)
        cv2.putText(frame, label, (x1+3, y1-5), self.font, self.font_scale, (255,255,255), self.thickness-1)

    def draw_scoreboard(self, frame, player_stats, top_n=10):
        """
        player_stats: dict of id -> distance
        """
        x, y = 10, 10
        lines = [f"Player Distances (m):"]
        # sort players by id or distance
        items = sorted(player_stats.items(), key=lambda x: x[0])
        for pid, dist in items[:top_n]:
            lines.append(f"ID {pid}: {dist:.1f} m")
        # draw background
        width = 260
        height = 20 * (len(lines) + 1)
        cv2.rectangle(frame, (x,y), (x+width, y+height), (50,50,50, 0.7), -1)
        # put text
        yy = y + 20
        for line in lines:
            cv2.putText(frame, line, (x+5, yy), self.font, 0.5, (255,255,255), 1)
            yy += 20
