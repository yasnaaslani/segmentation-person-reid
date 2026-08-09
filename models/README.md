# Model files

This demo expects two separately obtained model blobs:

1. A **416×416 YOLOv8n-seg COCO** DepthAI blob with outputs:
   - `output0`: `[1, 116, 3549]`
   - `output1`: `[1, 32, 104, 104]`
2. A person ReID blob accepting **128×256 BGR** person crops.

Development filenames:

```text
yolov8n-seg-416_openvino_2022.1_4shave.blob
person-reidentification-retail-0277.blob
```

The binary model files are intentionally excluded from Git. Obtain them separately and comply with their upstream licenses.
