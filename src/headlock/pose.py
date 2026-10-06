"""Single-person Halpe26 pose estimation (YOLOX detector + RTMPose) via rtmlib."""

from __future__ import annotations

import numpy as np

MODELS = {
    "rtmpose-halpe26-384": {
        "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "det_input_size": (640, 640),
        "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        "pose_input_size": (288, 384),  # (w, h)
    },
    "rtmpose-halpe26-256": {
        "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "det_input_size": (640, 640),
        "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "pose_input_size": (192, 256),
    },
}

# Our CLI device names -> rtmlib/onnxruntime device names.
DEVICES = {"cpu": "cpu", "coreml": "mps", "cuda": "cuda"}


def _area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _expand(box: np.ndarray, factor: float, w: int, h: int) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) * factor / 2, (box[3] - box[1]) * factor / 2
    return np.array([max(0, cx - hw), max(0, cy - hh), min(w, cx + hw), min(h, cy + hh)])


class BoxStabilizer:
    """Pick the dancer's box each frame and damp detector jitter.

    Picks the detection overlapping the previous box most (largest box on the first frame),
    smooths centre and size with an EMA, and pads ~10%. If the detector misses, the previous
    box is reused expanded by 10% so a fast move doesn't fall outside it.
    """

    def __init__(self, alpha: float = 0.6, pad: float = 1.1):
        self.alpha = alpha
        self.pad = pad
        self.prev: np.ndarray | None = None  # smoothed, unpadded box

    def update(self, boxes: np.ndarray, w: int, h: int) -> np.ndarray | None:
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if len(boxes) == 0:
            if self.prev is None:
                return None
            self.prev = _expand(self.prev, 1.1, w, h)
            return _expand(self.prev, self.pad, w, h)
        if self.prev is None:
            pick = boxes[np.argmax([_area(b) for b in boxes])]
            self.prev = pick
        else:
            pick = boxes[np.argmax([_iou(b, self.prev) for b in boxes])]
            self.prev = self.alpha * pick + (1 - self.alpha) * self.prev
        return _expand(self.prev, self.pad, w, h)


class PoseEstimator:
    def __init__(self, model: str = "rtmpose-halpe26-384", device: str = "cpu"):
        from rtmlib import RTMPose, YOLOX

        cfg = MODELS[model]
        dev = DEVICES[device]
        # CoreML can't run YOLOX's embedded NMS (onnxruntime static-shape error); keep it on CPU.
        det_dev = "cpu" if device == "coreml" else dev
        self.det = YOLOX(cfg["det"], model_input_size=cfg["det_input_size"],
                         backend="onnxruntime", device=det_dev)
        self.pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                            backend="onnxruntime", device=dev)
        self.boxes = BoxStabilizer()

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (26, 2) keypoints in frame pixels and (26,) scores. All-zero scores if no person."""
        h, w = frame.shape[:2]
        box = self.boxes.update(self.det(frame), w, h)
        if box is None:
            return np.zeros((26, 2)), np.zeros(26)
        kpts, scores = self.pose(frame, bboxes=[box.tolist()])
        return kpts[0], scores[0]
