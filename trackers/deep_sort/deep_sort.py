from .tracker import Tracker
from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from .generate_detections import create_box_encoder
import numpy as np

class DeepSort:
    def __init__(self, model_path, max_cosine_distance=0.2, nn_budget=100):
        self.encoder = create_box_encoder(model_path, batch_size=1)
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update_tracks(self, detections, frame):
        features = self.encoder(frame, [d[:4] for d in detections])
        dets = [Detection(bbox[:4], bbox[4], feat) for bbox, feat in zip(detections, features)]
        self.tracker.predict()
        self.tracker.update(dets)
        return self.tracker.tracks
