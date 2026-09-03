# DRIFT

Dynamic scheduling of multi-DNN inference on heterogeneous accelerators (GPU + DLA).

## Install

```bash
uv sync
```

## Public API

```python
from drift import DRIFT, AccuracyPredictor
```

- `DRIFT` — the scheduler (was `Dynamic` in the original `rewrite.py`).
- `AccuracyPredictor` — conf→F1 regression across models (was `Predictor`).

## Usage

```python
from pathlib import Path

from drift import DRIFT
from drift.paths import discover_engines
from drift.priorities import PEOPLE

gpu, dla0, dla1, _ = discover_engines(Path("/data/tensorrt_engines/orinagx/jetpack602"))

models = [p for tag, p in gpu if "10" in tag]
models += [p for tag, p in dla0 if "xs" in tag and ("480" in tag or "320" in tag)]

drift = DRIFT(
    models,
    PEOPLE,
    low_priority_accuracy_goal=0.5,
    highpri_pad=64,
    # all writable state is relocatable
    cache_path=Path("~/.cache/drift/cache.pickle").expanduser(),
    predictor_path=Path("~/.cache/drift/predictor.pickle").expanduser(),
    valdata_path=Path("~/.cache/drift/records/acc_f1_record.pickle").expanduser(),
)
```

Priority maps are `defaultdict(lambda: -1)` keyed by COCO class id, where `0` marks
high priority. `drift.priorities` ships `PEOPLE`, `VEHICLES`, and `ALL`.

## Engine discovery

`drift.paths.discover_engines(engine_root)` returns `(gpu, dla0, dla1, all)` lists of
`(tag, path)` pairs. It expects `engine_root` to contain per-family subdirectories
(`yolov7`, `yolov8`, `yolov9`, `yolov10`, `yolox`) holding engines named:

```
<model>_<imgsz>_<precision>_<gpu|dla0|dla1>.engine
```

A missing directory yields empty lists, so importing is always safe. The module-level
`YOLO_GPU_PATHS` / `YOLO_DLA0_PATHS` / `YOLO_DLA1_PATHS` / `YOLO_PATHS` are populated
from `DEFAULT_ENGINE_ROOT` (`<package>/../data/tensorrt_engines/orinagx/jetpack602`).

## Generating the accuracy records

`DRIFT(...)` cannot be constructed until an `acc_f1_record.pickle` exists — nothing is
vendored in this repo. To produce one:

1. Build TensorRT engines and lay them out as above.
2. Fetch COCO17 val, with `val/` and `annotations/instances_val2017.json`.
3. Run the sweep — this needs Jetson hardware and takes a long time:

   ```python
   from pathlib import Path
   from drift.acc_predictor import get_data

   get_data(
       record_dir=Path("~/.cache/drift/records").expanduser(),
       coco_dir=Path("/data/coco17"),
       engine_root=Path("/data/tensorrt_engines/orinagx/jetpack602"),
   )
   ```

   It runs every YOLOv10/YOLOX engine over COCO17 val and writes
   `acc_{f1,precision,recall}_record.pickle` into `record_dir`.

4. The first `DRIFT(...)` construction fits and caches `predictor.pickle`, and
   benchmarks each engine into `cache.pickle`.

> A `predictor.pickle` produced before this package existed will **not** load — it
> references the old `acc_predictor.Predictor` module path. Delete it and let it
> regenerate.

## Provenance

Migrated from `research/experiments/dac25/rewrite.py`. The port was deliberately
minimal: class/function renames, bare-name imports converted to package imports, and
hardcoded paths turned into arguments. Scheduler logic is unchanged.
