from __future__ import annotations

import cv2
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .i18n import LanguageManager

try:
    from pygrabber.dshow_graph import FilterGraph
except Exception:
    FilterGraph = None


def list_connected_cameras(
    max_tested: int = 10,
    i18n: LanguageManager | None = None
) -> List[Tuple[int, str]]:
    cameras: List[Tuple[int, str]] = []

    # DirectShow (Windows)
    if FilterGraph is not None:
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                cameras.append((idx, name))
            if cameras:
                return cameras
        except Exception:
            pass

    # Fallback OpenCV
    label_fmt = i18n.tr("camera_label_fmt", "Kamera {idx}") if i18n else "Kamera {idx}"

    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cameras.append((idx, label_fmt.format(idx=idx)))
        if cap is not None:
            cap.release()

    return cameras