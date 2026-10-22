import ast
import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import onnxruntime as ort
import torch  # noqa: F401 ; for CUDA DLLs


class Detection(TypedDict):
    cls: str  # class name
    score: int  # 0-100%
    box: tuple[int, int, int, int]  # (x, y, w, h)


class HaloCV:
    # TODO: could be offloaded to threads, if concurrency matters.
    # TODO: could add a model download helper.

    normalizer = np.float32(1.0 / 255.0)

    def __init__(
        self,
        model: str | Path,
        auto_resize: bool = False,
        log_meta: bool = False,
    ) -> None:

        self.auto_resize = auto_resize

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

        try:
            model_date = meta.get("date", "unknown")

            model_author = meta["author"]
            if model_author != "Ultralytics":
                raise RuntimeError(f"Unsupported author: {model_author}")

            model_task = meta["task"]
            if model_task != "detect":
                raise RuntimeError(f"Unsupported task: {model_task}")

            model_imgsz: list = ast.literal_eval(meta["imgsz"])
            self.resolution_h, self.resolution_w = model_imgsz

            self.classes: dict = ast.literal_eval(meta["names"])

        except KeyError as e:
            raise RuntimeError(f"Could not read model metadata: {e}")

        self.buffer_hwc = np.empty(
            (self.resolution_h, self.resolution_w, 3), dtype=np.uint8
        )  # buffer for letterboxing.

        self.buffer_nchw = np.empty(
            (1, 3, self.resolution_h, self.resolution_w), dtype=np.float32
        )  # buffer for normalized model input.

        if log_meta:
            meta_str = (
                f"Loaded {model}\n"
                f"Model Date: {model_date}\n"
                f"Model Resolution: (h: {self.resolution_h}, w: {self.resolution_w})\n"
                f"Model Classes:\n"
                f"{json.dumps(self.classes, indent=2)}"
            )
            print(meta_str)

    def _preprocess(
        self,
        frame: np.ndarray,
        result_resolution: tuple[int, int] | None,
    ) -> tuple[int, int, int, int, float, float]:

        h, w, c = frame.shape[:3]  # height, width, channels

        if c != 3:
            raise ValueError(
                f"frame channel mismatch, expected 3, received {c}.",
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                f"frame dtype mismatch, expected np.uint8, received {frame.dtype}.",
            )

        if h > self.resolution_h or w > self.resolution_w:
            if not self.auto_resize:
                raise ValueError(
                    f"frame resolution mismatch, expected (h: {self.resolution_h}, w: {self.resolution_w}), received (h: {h}, w: {w}).",
                )

            downscale = min(self.resolution_h / h, self.resolution_w / w, 1.0)

            w = max(1, round(w * downscale))
            h = max(1, round(h * downscale))

            frame = cv2.resize(
                frame,
                (w, h),
                interpolation=cv2.INTER_AREA,
            )

        ## -- 1: Letterboxing -- ##

        pad_h = self.resolution_h - h
        pad_w = self.resolution_w - w

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

        ## -- 2: Normalization -- ##

        np.multiply(
            buffer_hwc.transpose(2, 0, 1)[None, ...],
            self.normalizer,
            out=self.buffer_nchw,
        )

        if result_resolution is not None:
            target_w, target_h = result_resolution
            scale_x = w / target_w
            scale_y = h / target_h
        else:
            scale_x = scale_y = 1.0

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

        original_w = self.resolution_w - pad_left - pad_right
        original_h = self.resolution_h - pad_top - pad_bottom

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

            detections.append(
                {
                    "cls": self.classes[int(class_ids[i])],
                    "score": round(float(class_scores[i]) * 100),
                    "box": (int(x), int(y), int(w), int(h)),
                }
            )

        return detections

    def detect(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float,
        result_resolution: tuple[int, int] | None = None,
    ) -> list[Detection]:

        (
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            scale_x,
            scale_y,
        ) = self._preprocess(frame, result_resolution)

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
