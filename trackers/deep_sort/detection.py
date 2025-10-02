import numpy as np

class Detection:
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h])

    def to_xyah(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2., y + h / 2., w / float(h), h])
