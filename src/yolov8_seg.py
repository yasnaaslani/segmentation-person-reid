import cv2
import numpy as np


class YoloV8SegPersonParser:
    """Parser for YOLOv8n-seg COCO OpenVINO output.

    Expected 416x416 model outputs:
      output0: [1, 116, 3549] = 4 box + 80 classes + 32 mask coeffs
      output1: [1, 32, 104, 104] = mask prototypes

    Only COCO class 0 (person) is returned.
    """

    def __init__(
        self,
        input_size=416,
        conf_threshold=0.40,
        mask_threshold=0.50,
        max_raw_candidates=20,
        max_persons=8,
        mask_nms_iou=0.55,
        mask_containment=0.70,
    ):
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.mask_threshold = float(mask_threshold)
        self.max_raw_candidates = int(max_raw_candidates)
        self.max_persons = int(max_persons)
        self.mask_nms_iou = float(mask_nms_iou)
        self.mask_containment = float(mask_containment)

        # YOLOv8n-seg geometry for a 416x416 export.
        self.proto_size = self.input_size // 4
        self.mask_dim = 32
        self.pred_count = sum((self.input_size // s) ** 2 for s in (8, 16, 32))

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _xywh_to_xyxy(boxes):
        out = np.zeros_like(boxes, dtype=np.float32)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
        out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
        out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
        out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
        return out

    def _build_proto_mask(self, prototypes, coefficients, box_xyxy):
        logits = np.matmul(
            coefficients.astype(np.float32),
            prototypes.reshape(self.mask_dim, -1).astype(np.float32),
        ).reshape(self.proto_size, self.proto_size)
        binary = self._sigmoid(logits) > self.mask_threshold

        x1, y1, x2, y2 = box_xyxy
        p = self.proto_size
        s = self.input_size
        px1 = max(0, min(p, int(np.floor(x1 * p / s))))
        py1 = max(0, min(p, int(np.floor(y1 * p / s))))
        px2 = max(0, min(p, int(np.ceil(x2 * p / s))))
        py2 = max(0, min(p, int(np.ceil(y2 * p / s))))

        cropped = np.zeros((p, p), dtype=bool)
        if px2 > px1 and py2 > py1:
            cropped[py1:py2, px1:px2] = binary[py1:py2, px1:px2]
        return cropped

    @staticmethod
    def _mask_metrics(mask_a, mask_b):
        a = np.asarray(mask_a, dtype=bool)
        b = np.asarray(mask_b, dtype=bool)
        area_a = int(np.count_nonzero(a))
        area_b = int(np.count_nonzero(b))
        if area_a == 0 or area_b == 0:
            return 0.0, 0.0
        inter = int(np.count_nonzero(a & b))
        if inter == 0:
            return 0.0, 0.0
        union = area_a + area_b - inter
        iou = float(inter / union) if union > 0 else 0.0
        containment = float(inter / min(area_a, area_b))
        return iou, containment

    def _mask_nms(self, candidates):
        kept = []
        for candidate in candidates:
            duplicate = False
            for old in kept:
                mask_iou, containment = self._mask_metrics(
                    candidate["proto_mask"], old["proto_mask"]
                )
                if mask_iou >= self.mask_nms_iou or containment >= self.mask_containment:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
                if len(kept) >= self.max_persons:
                    break
        return kept

    def _mask_to_frame(self, proto_mask, model_box, orig_w, orig_h):
        low = proto_mask.astype(np.uint8) * 255
        model_mask = cv2.resize(
            low,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_NEAREST,
        )

        x1, y1, x2, y2 = model_box
        s = self.input_size
        x1i = max(0, min(s - 1, int(round(x1))))
        y1i = max(0, min(s - 1, int(round(y1))))
        x2i = max(0, min(s, int(round(x2))))
        y2i = max(0, min(s, int(round(y2))))
        hard = np.zeros_like(model_mask)
        if x2i > x1i and y2i > y1i:
            hard[y1i:y2i, x1i:x2i] = model_mask[y1i:y2i, x1i:x2i]

        return cv2.resize(hard, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def parse(self, nn_data, orig_w, orig_h):
        output0 = np.asarray(nn_data.getLayerFp16("output0"), dtype=np.float32)
        output1 = np.asarray(nn_data.getLayerFp16("output1"), dtype=np.float32)

        expected0 = 116 * self.pred_count
        expected1 = self.mask_dim * self.proto_size * self.proto_size
        if output0.size != expected0 or output1.size != expected1:
            raise RuntimeError(
                "Unexpected YOLOv8-seg output geometry. "
                f"output0={output0.size} expected={expected0}; "
                f"output1={output1.size} expected={expected1}. "
                "Use a YOLOv8n-seg blob compiled for the configured input size."
            )

        predictions = output0.reshape(1, 116, self.pred_count)[0].T
        prototypes = output1.reshape(
            1, self.mask_dim, self.proto_size, self.proto_size
        )[0]

        boxes_xywh = predictions[:, 0:4]
        class_scores = predictions[:, 4:84]
        mask_coeffs = predictions[:, 84:116]
        person_scores = class_scores[:, 0]

        raw_indices = np.where(person_scores >= self.conf_threshold)[0]
        if raw_indices.size == 0:
            return []

        order = np.argsort(person_scores[raw_indices])[::-1][: self.max_raw_candidates]
        raw_indices = raw_indices[order]
        model_boxes = self._xywh_to_xyxy(boxes_xywh[raw_indices])
        model_boxes[:, 0] = np.clip(model_boxes[:, 0], 0, self.input_size - 1)
        model_boxes[:, 1] = np.clip(model_boxes[:, 1], 0, self.input_size - 1)
        model_boxes[:, 2] = np.clip(model_boxes[:, 2], 0, self.input_size)
        model_boxes[:, 3] = np.clip(model_boxes[:, 3], 0, self.input_size)

        candidates = []
        for local_idx, raw_idx in enumerate(raw_indices):
            model_box = model_boxes[local_idx]
            proto_mask = self._build_proto_mask(
                prototypes, mask_coeffs[raw_idx], model_box
            )
            if np.count_nonzero(proto_mask) < 8:
                continue
            candidates.append(
                {
                    "model_box": model_box,
                    "proto_mask": proto_mask,
                    "score": float(person_scores[raw_idx]),
                }
            )

        kept = self._mask_nms(candidates)
        scale_x = float(orig_w) / float(self.input_size)
        scale_y = float(orig_h) / float(self.input_size)

        people = []
        for item in kept:
            x1, y1, x2, y2 = item["model_box"]
            frame_box = (
                max(0, min(orig_w - 1, int(round(x1 * scale_x)))),
                max(0, min(orig_h - 1, int(round(y1 * scale_y)))),
                max(0, min(orig_w, int(round(x2 * scale_x)))),
                max(0, min(orig_h, int(round(y2 * scale_y)))),
            )
            if frame_box[2] <= frame_box[0] or frame_box[3] <= frame_box[1]:
                continue
            people.append(
                {
                    "bbox": frame_box,
                    "mask": self._mask_to_frame(
                        item["proto_mask"], item["model_box"], orig_w, orig_h
                    ),
                    "score": item["score"],
                }
            )
        return people
