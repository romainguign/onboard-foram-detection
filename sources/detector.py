import cv2


class YoloDetector:
    """YOLO wrapper returning an annotated frame and structured detections."""

    def __init__(self, model, classifier=None, crop_margin=0):
        self.model = model
        self.classifier = classifier
        self.crop_margin = crop_margin

    def _crop_with_margin(self, frame, x1, y1, x2, y2):
        """Crop a box with optional margin, clamped inside image bounds."""
        h, w = frame.shape[:2]

        x1m = max(0, x1 - self.crop_margin)
        y1m = max(0, y1 - self.crop_margin)
        x2m = min(w, x2 + self.crop_margin)
        y2m = min(h, y2 + self.crop_margin)

        if x2m <= x1m or y2m <= y1m:
            return None

        return frame[y1m:y2m, x1m:x2m]

    def predict(self, frame):
        """
        Run YOLO detection on the full frame.
        If a classifier is available, batch-classify all crops in one call.
        """
        results = self.model(frame, verbose=False)
        annotated = frame.copy()
        detections = []

        # Temporary storage for batch classification
        crops = []
        crop_indices = []

        # First pass: collect detections and crops
        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0

                detected_label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                detection = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detected_label": detected_label,
                    "confidence": conf,
                    "editable_label": "unknown",
                    "classification_score": None,
                }
                detections.append(detection)

                if self.classifier is not None:
                    crop = self._crop_with_margin(frame, x1, y1, x2, y2)
                    if crop is not None and crop.size > 0:
                        crops.append(crop)
                        crop_indices.append(len(detections) - 1)

        # Batch classification
        if self.classifier is not None and crops:
            batch_results = self.classifier.predict_batch(crops)

            for det_idx, cls_result in zip(crop_indices, batch_results):
                class_name, score, class_index = cls_result
                detections[det_idx]["editable_label"] = class_name
                detections[det_idx]["classification_score"] = score

        # Second pass: draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            if det["classification_score"] is not None:
                text = f"{det['editable_label']} {det['classification_score']:.2f}"
            else:
                text = f"{det['detected_label']} {det['confidence']:.2f}"

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

        return annotated, detections