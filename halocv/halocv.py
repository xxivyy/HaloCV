import ast
import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import onnxruntime as ort
import torch  # noqa: F401 ; for CUDA DLLs


class Detection(TypedDict):
    """A single detected object."""

    cls: str  # class label.
    score: int  # 0-100 confidence score.
    box: tuple[int, int, int, int]  # (x, y, width, height)


class HaloCV:
    """
    Object detector for Ultralytics YOLO `.onnx` models.

    Args:
        model: path to `.onnx` model.
        log_meta: whether to print model metadata during initialization.

    Attributes:
        input_height: model input height.
        input_width: model input width.
        classes: dictionary mapping class IDs to class names.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model file is not a `.onnx` file.
        RuntimeError: If the model metadata is missing or unsupported.
    """

    # TODO: could implement a threadpool, if concurrency matters.

    normalizer = np.float32(1.0 / 255.0)

    def __init__(
        self,
        model: str | Path,
        log_meta: bool = False,
    ) -> None:

        model = Path(model).resolve()

        if not model.exists():
            raise FileNotFoundError(f"Model file not found: {model}")

        if model.suffix != ".onnx":
            raise ValueError(f"Model file must be a .onnx file: {model}")

        self.session = ort.InferenceSession(
            model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        meta: dict = self.session.get_modelmeta().custom_metadata_map

        model_date = meta.get("date", "unknown")

        try:
            model_author = meta["author"]
            if model_author != "Ultralytics":
                raise RuntimeError(f"Unsupported author: {model_author}")

            model_task = meta["task"]
            if model_task != "detect":
                raise RuntimeError(f"Unsupported task: {model_task}")

            model_imgsz: list = ast.literal_eval(meta["imgsz"])

            self.input_height, self.input_width = model_imgsz

            self.classes: dict = ast.literal_eval(meta["names"])

        except KeyError as e:
            raise RuntimeError(f"Could not read model metadata: {e}")

        self.buffer_hwc = np.empty(
            (self.input_height, self.input_width, 3), dtype=np.uint8
        )  # buffer for letterboxing.

        self.buffer_nchw = np.empty(
            (1, 3, self.input_height, self.input_width), dtype=np.float32
        )  # buffer for normalized model input.

        if log_meta:
            meta_str = (
                f"Loaded {model}\n"
                f"Model Date: {model_date}\n"
                f"Model Resolution: (h: {self.input_height}, w: {self.input_width})\n"
                f"Model Classes:\n"
                f"{json.dumps(self.classes, indent=2)}"
            )
            print(meta_str)

    def _preprocess(
        self,
        frame: np.ndarray,
        box_resolution: tuple[int, int] | None,
    ) -> tuple[int, int, int, int, float, float]:

        ## -- 1: Validation -- ##

        if frame.ndim != 3:
            raise ValueError(
                f"frame dimension mismatch, expected 3, received {frame.ndim}."
            )

        if frame.shape[2] != 3:  # color channels
            raise ValueError(
                f"frame channel mismatch, expected 3, received {frame.shape[2]}."
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                f"frame dtype mismatch, expected np.uint8, received {frame.dtype}.",
            )

        frame_h, frame_w = frame.shape[:2]

        if frame_h == 0 or frame_w == 0:
            raise ValueError(
                f"Invalid frame shape: (h: {frame_h}, w: {frame_w}). Both dimensions must be positive."
            )

        if box_resolution is not None:
            target_w, target_h = box_resolution

            if target_w <= 0 or target_h <= 0:
                raise ValueError(
                    f"Invalid box_resolution: {box_resolution}. Both values must be positive.",
                )

        ## -- 2: Resizing -- ##

        if frame_h > self.input_height or frame_w > self.input_width:
            scale = min(1.0, self.input_height / frame_h, self.input_width / frame_w)

            w = max(1, round(frame_w * scale))
            h = max(1, round(frame_h * scale))

            frame = cv2.resize(
                src=frame,
                dsize=(w, h),
                interpolation=cv2.INTER_AREA,
            )

        else:
            h, w = frame_h, frame_w

        ## -- 3: Letterboxing -- ##

        pad_h = self.input_height - h
        pad_w = self.input_width - w

        if pad_h > 0 or pad_w > 0:
            left = pad_w // 2
            right = pad_w - left

            top = pad_h // 2
            bottom = pad_h - top

            cv2.copyMakeBorder(
                src=frame,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                borderType=cv2.BORDER_CONSTANT,
                dst=self.buffer_hwc,
                value=(114, 114, 114),
            )

            buffer_hwc = self.buffer_hwc

        else:  # No padding needed.
            left = right = top = bottom = 0
            buffer_hwc = frame

        ## -- 4: Normalization -- ##

        np.multiply(
            buffer_hwc.transpose(2, 0, 1)[None, ...],
            self.normalizer,
            out=self.buffer_nchw,
        )

        ## -- 5: Scaling -- ##

        if box_resolution is not None:
            scale_x = w / target_w
            scale_y = h / target_h

        else:  # scale to original frames resolution.
            scale_x = w / frame_w
            scale_y = h / frame_h

        return left, right, top, bottom, scale_x, scale_y

    def _postprocess(
        self,
        output: np.ndarray,
        pad_left: int,
        pad_right: int,
        pad_top: int,
        pad_bottom: int,
        scale_x: float,
        scale_y: float,
        conf: float,
        iou: float,
    ) -> list[Detection]:

        ## -- 1: Parse model output -- ##

        # Drop d1 and swap d2/d3.
        # Column: prediction ; Rows: boxes, scores for each class.

        prediction = output.squeeze(0).T

        boxes = prediction[:, :4]
        scores = prediction[:, 4:]

        # Get the class ids (indices) with the highest scores for each prediction.

        class_ids = np.argmax(scores, axis=1)
        class_scores = scores[np.arange(len(class_ids)), class_ids]

        # Create a bool mask to only keep predictions with scores above the conf threshold.

        keep_mask = class_scores >= conf

        boxes = boxes[keep_mask]
        class_ids = class_ids[keep_mask]
        class_scores = class_scores[keep_mask]

        if len(boxes) == 0:
            return []

        ## -- 2: Clip boxes to image bounds -- ##

        # convert boxes: cxcywh -> xyxy

        boxes_xyxy = np.empty_like(boxes)

        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 left
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 top
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 right
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 bottom

        # revert letterbox padding

        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top

        original_w = self.input_width - pad_left - pad_right
        original_h = self.input_height - pad_top - pad_bottom

        # clip to original image bounds

        np.clip(
            boxes_xyxy[:, 0::2],
            0,
            original_w,
            out=boxes_xyxy[:, 0::2],
        )

        np.clip(
            boxes_xyxy[:, 1::2],
            0,
            original_h,
            out=boxes_xyxy[:, 1::2],
        )

        ## -- 3: Apply non maxium suppression (NMS) -- ##

        # convert boxes: xyxy -> xywh

        boxes_xyxy[:, 2] -= boxes_xyxy[:, 0]
        boxes_xyxy[:, 3] -= boxes_xyxy[:, 1]
        boxes_xywh = boxes_xyxy

        # apply nms ; get the indices of the remaining boxes

        indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(),
            class_scores.tolist(),
            class_ids.tolist(),
            conf,
            iou,
        )

        if len(indices) == 0:
            return []

        ## -- 4: Summarize final detections -- ##

        detections: list[Detection] = []

        for i in np.asarray(indices).reshape(-1):
            x, y, w, h = boxes_xywh[i]

            x1 = round(float(x) / scale_x)
            y1 = round(float(y) / scale_y)
            x2 = round(float(x + w) / scale_x)
            y2 = round(float(y + h) / scale_y)

            detections.append(
                {
                    "cls": self.classes[int(class_ids[i])],
                    "score": round(float(class_scores[i]) * 100),
                    "box": (x1, y1, x2 - x1, y2 - y1),
                }
            )

        return detections

    def detect(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float,
        box_resolution: tuple[int, int] | None = None,
    ) -> list[Detection]:
        """
        Detect objects in the given frame.

        Args:
            frame: Input `RGB` `uint8` image with shape `(height, width, 3)`.
                Frames larger than the model input are downscaled internally.
                Make sure to convert to `RGB` first if necessary, for example when using a `cv2.imread` (`BGR`).
            conf: `0-1` Confidence threshold.
            iou: `0-1` Intersection over Union (IoU) threshold for NMS.
            box_resolution: The `(width, height)` to which the returned boxes should be scaled to.
                For example if you resized the passed `frame`, but need the coordinates for the original image.
                If `None`, the box coordinates are based on the originally passed `frame`'s resolution.

        Returns:
            `list[dict]`: List of detected objects, with the following keys:
            `"cls": str` class label of the detected object.
            `"score": int` 0-100 confidence score.
            `"box": tuple[int, int, int, int]` bounding box coordinates `(x, y, width, height)`.

        Raises:
            ValueError: If `frame` has an unexpected shape or type.
            ValueError: If `box_resolution` is provided but has non-positive dimensions.
        """

        (
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            scale_x,
            scale_y,
        ) = self._preprocess(frame, box_resolution)

        output = self.session.run(
            None,
            {self.input_name: self.buffer_nchw},
        )[0]

        result = self._postprocess(
            output,  # ty: ignore[invalid-argument-type]
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            scale_x,
            scale_y,
            conf,
            iou,
        )

        return result
