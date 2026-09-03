# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

DEFAULT_ENGINE_ROOT = (
    Path(__file__).parent.parent / "data" / "tensorrt_engines" / "orinagx" / "jetpack602"
)


def discover_engines(
    engine_root: Path = DEFAULT_ENGINE_ROOT,
) -> tuple[
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path]],
]:
    """Discover TensorRT engines under engine_root.

    Returns (gpu, dla0, dla1, all) lists of (tag, path) pairs.
    Missing directories yield empty lists.
    """
    gpu_paths: list[tuple[str, Path]] = []
    dla0_paths: list[tuple[str, Path]] = []
    dla1_paths: list[tuple[str, Path]] = []
    all_paths: list[tuple[str, Path]] = []

    yolo_dirs = [
        engine_root / "yolov7",
        engine_root / "yolov8",
        engine_root / "yolov9",
        engine_root / "yolov10",
        engine_root / "yolox",
    ]
    for ydir in yolo_dirs:
        gpu_engines = sorted(
            list(ydir.glob("*gpu.engine")),
            key=lambda e: int(e.stem.split("_")[1]),
        )
        dla0_engines = sorted(
            list(ydir.glob("*dla0.engine")),
            key=lambda e: int(e.stem.split("_")[1]),
        )
        dla0_engines = [e for e in dla0_engines if "fp16" not in e.stem]
        dla1_engines = sorted(
            list(ydir.glob("*dla1.engine")),
            key=lambda e: int(e.stem.split("_")[1]),
        )
        dla1_engines = [e for e in dla1_engines if "fp16" not in e.stem]

        for engines, paths in zip(
            [gpu_engines, dla0_engines, dla1_engines],
            [gpu_paths, dla0_paths, dla1_paths],
        ):
            for eng in engines:
                model, imgsz, _, device = eng.stem.split("_")
                if "yolov" in model:
                    vtag = f"{model.replace('yolov', '')}_{imgsz}"
                else:
                    vtag = f"{model.replace('yolo', '')}_{imgsz}"
                paths.append((vtag, eng))
                global_vtag = f"{vtag}_{device}"
                all_paths.append((global_vtag, eng))

    return gpu_paths, dla0_paths, dla1_paths, all_paths


YOLO_GPU_PATHS, YOLO_DLA0_PATHS, YOLO_DLA1_PATHS, YOLO_PATHS = discover_engines()
