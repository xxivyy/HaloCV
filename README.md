<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-brightgreen)](https://www.python.org/)
[![Inference](https://img.shields.io/badge/Inference-ONNX_Runtime-orange)](https://pypi.org/project/onnxruntime/)
[![Models](https://img.shields.io/badge/Models-Ultralytics_YOLO-blue)](https://github.com/ultralytics/ultralytics)
[![GUI](https://img.shields.io/badge/GUI-PySide6-magenta)](https://pypi.org/project/PySide6/)
[![Capture](https://img.shields.io/badge/Capture-DxCam-magenta)](https://github.com/ra1nty/DXcam)
[![License](https://img.shields.io/badge/License-AGPLv3-yellow)](https://github.com/xxivyy/HaloCV/blob/main/LICENSE)

### [Quickstart](#quickstart) • [About](#about) • [Usage](#usage) • [License](#license)

</div>

## Quickstart

```python
import cv2

from halocv import HaloCV

model = HaloCV("path/to/model.onnx")

image = cv2.imread("path/to/image.png")

if image is not None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = model.detect(image, conf=0.55, iou=0.45)

    for d in detections:
        print(d)
```

## About

...

## Usage

...

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
