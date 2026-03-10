from __future__ import annotations

import cv2


def configure_runtime(opencv_threads: int = 1) -> None:
    try:
        cv2.setNumThreads(max(1, int(opencv_threads)))
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
