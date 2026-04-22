import cv2


class YoloDetector:
    """YOLO wrapper returning an annotated frame and structured detections."""

    def __init__(self, model):
        self.model = model

    def predict(self, frame):
        """
        Run inference on a BGR OpenCV frame.

        Returns:
            annotated_frame: frame with boxes and labels drawn
            detections: list of dicts
        """
        results = self.model(frame, verbose=False)
        annotated = frame.copy()
        detections = []

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0

                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                text = f"{label} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "detected_label": label,
                        "confidence": conf,
                        "editable_label": "unknown",
                    }
                )

        return annotated, detections