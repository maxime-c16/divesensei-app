from __future__ import annotations

import cv2
import os


def configure_runtime(opencv_threads: int = 0) -> None:
    threads = int(opencv_threads)
    if threads <= 0:
        threads = int(os.cpu_count() or 1)
    try:
        cv2.setNumThreads(max(1, threads))
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
