import argparse
import os
import time

import cv2
import depthai as dai
import numpy as np

from reid import cosine_similarity, normalize_embedding, tight_mask_crop, to_planar_imgframe
from yolov8_seg import YoloV8SegPersonParser


class ClickState:
    def __init__(self):
        self.point = None

    def callback(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)


def build_pipeline(seg_model, reid_model, input_size, fps):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(416, 240)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(float(fps))

    seg_manip = pipeline.create(dai.node.ImageManip)
    seg_manip.initialConfig.setResize(input_size, input_size)
    seg_manip.initialConfig.setKeepAspectRatio(False)
    seg_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    seg_manip.setMaxOutputFrameSize(input_size * input_size * 3)
    cam.preview.link(seg_manip.inputImage)

    seg_nn = pipeline.create(dai.node.NeuralNetwork)
    seg_nn.setBlobPath(seg_model)
    seg_nn.setNumInferenceThreads(1)
    seg_nn.input.setBlocking(True)
    seg_nn.input.setQueueSize(8)
    seg_manip.out.link(seg_nn.input)

    rgb_out = pipeline.create(dai.node.XLinkOut)
    rgb_out.setStreamName("rgb")
    cam.preview.link(rgb_out.input)

    seg_out = pipeline.create(dai.node.XLinkOut)
    seg_out.setStreamName("seg")
    seg_nn.out.link(seg_out.input)

    # Person ReID is intentionally fed with host-created segmentation-only crops.
    reid_in = pipeline.create(dai.node.XLinkIn)
    reid_in.setStreamName("reid_in")

    reid_nn = pipeline.create(dai.node.NeuralNetwork)
    reid_nn.setBlobPath(reid_model)
    reid_nn.setNumInferenceThreads(2)
    reid_in.out.link(reid_nn.input)

    reid_out = pipeline.create(dai.node.XLinkOut)
    reid_out.setStreamName("reid_out")
    reid_nn.out.link(reid_out.input)

    return pipeline


def point_person_index(point, people):
    if point is None:
        return None
    x, y = point
    # Prefer actual mask ownership. Fall back to no selection rather than bbox-only ownership.
    for idx, person in enumerate(people):
        mask = person["mask"]
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            return idx
    return None


def overlay(frame, people, similarities, reference_index=None):
    out = frame.copy()
    tint = frame.copy()
    for person in people:
        mask = person["mask"] > 0
        tint[mask] = (255, 255, 255)
    out = cv2.addWeighted(tint, 0.30, out, 0.70, 0)

    for idx, person in enumerate(people):
        x1, y1, x2, y2 = person["bbox"]
        score = similarities[idx] if idx < len(similarities) else None
        label = f"P{idx + 1}"
        if score is not None:
            label += f"  sim={score:.3f}"
        if reference_index == idx:
            label += "  [REFERENCE]"

        contours, _ = cv2.findContours(
            (person["mask"] > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (220, 220, 220), 1)
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Segmentation-masked person ReID on OAK-D. Click a person to enroll a reference."
    )
    ap.add_argument("--seg-model", required=True, help="YOLOv8n-seg 416x416 .blob")
    ap.add_argument("--reid-model", required=True, help="Person ReID .blob, e.g. retail-0277")
    ap.add_argument("--input-size", type=int, default=416)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--mask-threshold", type=float, default=0.50)
    ap.add_argument("--warmup", type=float, default=1.5)
    args = ap.parse_args()

    for path in (args.seg_model, args.reid_model):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    parser = YoloV8SegPersonParser(
        input_size=args.input_size,
        conf_threshold=args.conf,
        mask_threshold=args.mask_threshold,
    )
    pipeline = build_pipeline(
        args.seg_model, args.reid_model, args.input_size, args.fps
    )

    click = ClickState()
    window = "Segmentation Person ReID | click=enroll | r=reset | q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, click.callback)

    reference_embedding = None
    reference_preview = None

    with dai.Device(pipeline) as device:
        print(f"USB speed: {device.getUsbSpeed()}")
        rgb_q = device.getOutputQueue("rgb", maxSize=8, blocking=True)
        seg_q = device.getOutputQueue("seg", maxSize=8, blocking=True)
        reid_in_q = device.getInputQueue("reid_in")
        reid_out_q = device.getOutputQueue("reid_out", maxSize=16, blocking=True)

        started = time.time()

        while True:
            frame = rgb_q.get().getCvFrame()
            seg_msg = seg_q.get()
            h, w = frame.shape[:2]
            warmup = time.time() - started < args.warmup
            people = [] if warmup else parser.parse(seg_msg, w, h)

            # Produce one segmentation-only ReID embedding for every visible person.
            embeddings = []
            valid_map = []
            for idx, person in enumerate(people):
                crop = tight_mask_crop(frame, person["mask"])
                if crop is None:
                    embeddings.append(None)
                    continue
                reid_in_q.send(to_planar_imgframe(crop))
                valid_map.append(idx)
                embeddings.append("pending")

            for idx in valid_map:
                result = reid_out_q.get()
                emb = normalize_embedding(result.getFirstLayerFp16())
                embeddings[idx] = emb

            # Clicking the actual segmentation mask enrolls exactly that visible instance.
            selected = point_person_index(click.point, people)
            click.point = None
            if selected is not None and selected < len(embeddings):
                emb = embeddings[selected]
                if isinstance(emb, np.ndarray):
                    reference_embedding = emb.copy()
                    reference_preview = tight_mask_crop(frame, people[selected]["mask"])
                    print(f"[ENROLL] person {selected + 1} selected as reference")

            similarities = []
            for emb in embeddings:
                if reference_embedding is None or not isinstance(emb, np.ndarray):
                    similarities.append(None)
                else:
                    similarities.append(cosine_similarity(reference_embedding, emb))

            reference_idx = None
            if reference_embedding is not None:
                valid_scores = [
                    (i, s) for i, s in enumerate(similarities) if s is not None
                ]
                if valid_scores:
                    reference_idx = max(valid_scores, key=lambda x: x[1])[0]

            vis = overlay(frame, people, similarities, reference_idx)
            status = "CAMERA WARMUP" if warmup else (
                "Click a person mask to enroll" if reference_embedding is None else "Reference enrolled"
            )
            cv2.putText(
                vis,
                status,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if reference_preview is not None:
                thumb = cv2.resize(reference_preview, (64, 128))
                y2 = min(vis.shape[0], 10 + thumb.shape[0])
                x2 = min(vis.shape[1], 10 + thumb.shape[1])
                vis[10:y2, 10:x2] = thumb[: y2 - 10, : x2 - 10]
                cv2.putText(
                    vis,
                    "REF",
                    (12, min(vis.shape[0] - 5, 150)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                reference_embedding = None
                reference_preview = None
                print("[RESET] reference cleared")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
