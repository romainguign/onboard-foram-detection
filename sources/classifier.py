import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import onnxruntime as ort


def load_from_xml(filename):
    project = ET.parse(filename).getroot()

    protobuf = project.find("protobuf").text

    img_size = np.zeros(3, dtype=int)
    cls_labels = []

    labels_xml = project.find("labels")
    for entry_xml in labels_xml.iter("label"):
        code = entry_xml.find("code").text
        cls_labels.append(code)

    inputs_xml = project.find("inputs")
    input_name_from_xml = None
    for i, entry_xml in enumerate(inputs_xml.iter("input")):
        if i == 0:
            input_name_from_xml = entry_xml.find("operation").text
            img_size[0] = int(entry_xml.find("height").text)
            img_size[1] = int(entry_xml.find("width").text)
            img_size[2] = int(entry_xml.find("channels").text)

    outputs_xml = project.find("outputs")
    output_name_from_xml = None
    for i, entry_xml in enumerate(outputs_xml.iter("output")):
        if i == 0:
            output_name_from_xml = entry_xml.find("operation").text

    cnn_tag = project.find(".//params/cnn")
    img_type = "rgb"
    if cnn_tag is not None and cnn_tag.text:
        txt = cnn_tag.text
        if "'k'" in txt or "'greyscale'" in txt:
            img_type = "k"

    full_model_path = os.path.join(os.path.dirname(filename), protobuf)

    return full_model_path, img_size, img_type, cls_labels, input_name_from_xml, output_name_from_xml


class ForamClassifier:
    """Classifier wrapper for cropped foram images using ONNX Runtime."""

    def __init__(self, model_info_path, unsure_threshold=0.5):
        (
            model_path,
            self.img_size,
            self.img_type,
            self.labels,
            self.input_name_from_xml,
            self.output_name_from_xml,
        ) = load_from_xml(model_info_path)

        self.unsure_threshold = unsure_threshold

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"[Classifier] ONNX input name: {self.input_name}")
        print(f"[Classifier] ONNX output name: {self.output_name}")
        print(f"[Classifier] Expected image size: {self.img_size}")
        print(f"[Classifier] ONNX input shape: {self.input_shape}")

    def preprocess_crop(self, crop):
        """
        Convert an OpenCV crop (BGR) to a single model input sample.
        Returns one sample without batch dimension:
            - HWC for NHWC models
            - CHW for NCHW models
        """
        if crop is None or crop.size == 0:
            return None

        target_h, target_w, target_c = self.img_size

        if target_c == 1:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (target_w, target_h))
            resized = resized.astype(np.float32) / 255.0
            resized = np.expand_dims(resized, axis=-1)  # HWC
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_w, target_h))
            resized = resized.astype(np.float32) / 255.0  # HWC

        if len(self.input_shape) != 4:
            raise ValueError(f"Unexpected ONNX input shape: {self.input_shape}")

        # NHWC
        if self.input_shape[1] == target_h and self.input_shape[2] == target_w:
            return resized.astype(np.float32)

        # NCHW
        if self.input_shape[1] == target_c and self.input_shape[2] == target_h:
            return np.transpose(resized, (2, 0, 1)).astype(np.float32)

        # Fallback
        return resized.astype(np.float32)

    def _decode_prediction(self, pred):
        class_index = int(np.argmax(pred))
        score = float(np.max(pred))

        if score < self.unsure_threshold:
            return "unsure", score, -1

        class_name = self.labels[class_index]
        return class_name, score, class_index

    def predict(self, crop):
        """Predict a single crop."""
        sample = self.preprocess_crop(crop)
        if sample is None:
            return "invalid_crop", 0.0, -1

        batch = np.expand_dims(sample, axis=0).astype(np.float32)

        preds = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )[0]

        return self._decode_prediction(preds[0])

    def predict_batch(self, crops):
        """
        Predict multiple crops in one inference call.

        Returns a list with the same length as `crops`.
        Each item is:
            (class_name, score, class_index)
        """
        if not crops:
            return []

        batch_samples = []
        valid_indices = []

        for i, crop in enumerate(crops):
            sample = self.preprocess_crop(crop)
            if sample is None:
                continue

            batch_samples.append(sample)
            valid_indices.append(i)

        if not batch_samples:
            return [("invalid_crop", 0.0, -1) for _ in crops]

        batch = np.stack(batch_samples, axis=0).astype(np.float32)

        preds = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )[0]

        results = [("invalid_crop", 0.0, -1) for _ in crops]

        for pred_idx, crop_idx in enumerate(valid_indices):
            results[crop_idx] = self._decode_prediction(preds[pred_idx])

        return results