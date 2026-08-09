import cv2
import depthai as dai
import numpy as np


def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-12:
        return None
    return emb / norm


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def tight_mask_crop(frame, mask, pad_ratio=0.04):
    """Create a ReID crop from segmentation pixels only.

    Crop geometry comes from the mask, not from the detector bbox. Every pixel
    outside the person mask is black before the crop is resized to 128x256.
    """
    binary = np.asarray(mask) > 0
    ys, xs = np.nonzero(binary)
    if xs.size < 20:
        return None

    h, w = frame.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = max(2, int(round(bw * pad_ratio)))
    py = max(2, int(round(bh * pad_ratio)))
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    segmented = np.zeros_like(frame)
    segmented[binary] = frame[binary]
    crop = segmented[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)


def to_planar_imgframe(bgr_128x256):
    if bgr_128x256.shape[:2] != (256, 128):
        raise ValueError("ReID input must be 128x256 BGR")
    msg = dai.ImgFrame()
    msg.setType(dai.RawImgFrame.Type.BGR888p)
    msg.setWidth(128)
    msg.setHeight(256)
    msg.setData(bgr_128x256.transpose(2, 0, 1).flatten())
    return msg
