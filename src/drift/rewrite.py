# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: F841
from __future__ import annotations

import contextlib
import math
import pickle
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import cv2ext
import numpy as np
from cv2ext.image.color import Color
from jetsontools import Tegrastats, filter_data, get_powerdraw, parse_tegrastats
from trtutils import jetson, set_log_level
from trtutils.impls.yolo import YOLO, ParallelYOLO

if TYPE_CHECKING:
    from typing_extensions import Self

from cv2ext.detection import AnnealingFramePacker

from drift.acc_predictor import AccuracyPredictor


class Timer:
    def __init__(self, tag: str):
        self._tag = tag
        self._t0: float
        self._t1: float

    def __enter__(self) -> self:
        self._t0 = time.time()
        return self

    def __exit__(self, a1, a2, a3) -> None:
        self._t1 = time.time()
        print(f"{self._tag}: {round((self._t1 - self._t0) * 1000.0, 2)} ms")


COCO_LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "street sign",
    12: "stop sign",
    13: "parking meter",
}

# set the priorities
PRIORITIES = defaultdict(lambda: -1)
# car, motorcycle, bus, truck
PRIORITIES[2] = 0
PRIORITIES[3] = 0
PRIORITIES[5] = 0
PRIORITIES[7] = 0
# # person, bicycle
# PRIORITIES[0] = 0
# PRIORITIES[1] = 0


from typing_extensions import TypeAlias

RatioPadding: TypeAlias = "tuple[tuple[float, float], tuple[float, float]]"
PreprocessPacket: TypeAlias = (
    "tuple[np.ndarray, tuple[float, float], tuple[float, float]]"
)
Output: TypeAlias = "tuple[list[np.ndarray], tuple[float, float], tuple[float, float]]"
Detections: TypeAlias = "list[tuple[tuple[int, int, int, int], float, int]]"


def benchmark_yolo(yolo: YOLO) -> tuple[tuple[float, float, float], float, float]:
    result = jetson.benchmark_engine(yolo.engine, tegra_interval=1)
    infer_data = (
        result.latency.mean * 1000.0,
        result.power_draw.mean / 1000.0,
        result.energy.mean / 1000.0,
    )
    preproc_data = {}

    rand_data: np.ndarray = np.random.default_rng().integers(
        0,
        255,
        (1080, 1920, 3),
        dtype=np.uint8,
    )

    for pre_method in ["cpu", "cuda"]:
        timestamps = []
        iters = 100 if pre_method == "cuda" else 10
        for _ in range(iters):
            t0 = time.time()
            yolo.preprocess(rand_data, resize="letterbox", method=pre_method)
            t1 = time.time()
            timestamps.append((t0, t1))
        latency_data = [t1 - t0 for (t0, t1) in timestamps]
        avg_latency: float = float(np.mean(latency_data)) * 1000.0
        preproc_data[pre_method] = avg_latency

    packet = infer_data, preproc_data["cpu"], preproc_data["cuda"]
    packet_str = f"(({packet[0][0]:.2f} ms, {packet[0][1]:.2f} W, {packet[0][2]:.2f} J), {packet[1]:.2f} ms, {packet[2]:.2f} ms)"
    print(f"Benched: {yolo.name} -> {packet_str}")
    return packet


def benchmark_parallel_yolo(
    pyolo: ParallelYOLO, gpu_id: int, dla_id: int,
) -> tuple[tuple[float, float, float], tuple[float, float], float, float]:
    rand_data: np.ndarray = np.random.default_rng().integers(
        0,
        255,
        (1080, 1920, 3),
        dtype=np.uint8,
    )

    timestamps: list[tuple[float, float, float, float, float, float]] = []
    tegra_file: Path = Path("temptegra.txt")
    with Tegrastats(tegra_file, interval=1):
        for _ in range(1000):
            t0 = time.time()
            gpu_tensor, gpu_ratio, gpu_padding = pyolo.preprocess_model(
                rand_data, gpu_id, resize="linear", method="cuda", no_copy=True,
            )
            pyolo.submit_model(
                gpu_tensor, gpu_id, preprocessed=True, postprocess=False, no_copy=True,
            )
            t1 = time.time()
            dla_tensor, dla_ratio, dla_padding = pyolo.preprocess_model(
                rand_data, dla_id, resize="letterbox", method="cuda", no_copy=True,
            )
            pyolo.submit_model(
                dla_tensor, dla_id, preprocessed=True, postprocess=False, no_copy=True,
            )
            t2 = time.time()
            gpu_output = pyolo.retrieve_model(gpu_id)
            t3 = time.time()
            dla_output = pyolo.retrieve_model(dla_id)
            t4 = time.time()
            timestamps.append((t0, t1, t2, t3, t4))

            # gpu_preproc = t1 - t0
            # dla_preproc = t2 - t1
            # gpu_infer = t3 - t1
            # dla_infer = t4 - t2
            # overall_infer = t4 - t1

    infer_timestamps = [(t1, t4) for (_, t1, _, _, t4) in timestamps]
    latencies = [t1 - t0 for t0, t1 in infer_timestamps]
    gpu_latencies = [t3 - t1 for _, t1, _, t3, _ in timestamps]
    dla_latencies = [t4 - t2 for _, _, t2, _, t4 in timestamps]
    avg_gpu_lat = float(np.mean(gpu_latencies)) * 1000.0
    avg_dla_lat = float(np.mean(dla_latencies)) * 1000.0
    gpu_preproc = [t1 - t0 for t0, t1, _, _, _ in timestamps]
    avg_gpu_preproc = float(np.mean(gpu_preproc)) * 1000.0
    dla_preproc = [t2 - t1 for _, t1, t2, _, _ in timestamps]
    avg_dla_preproc = float(np.mean(dla_preproc)) * 1000.0

    tegradata = parse_tegrastats(tegra_file)
    tegra_file.unlink()
    avg_latency = float(np.mean(latencies)) * 1000.0
    filtered, _ = filter_data(tegradata, infer_timestamps)
    # compute the average power draw first
    avg_powerdraw = get_powerdraw(filtered)["VDD_TOTAL"].mean
    avg_powerdraw /= 1000.0  # convert to W
    # iterate over the per_timestamp data to build powerdraw and energy
    energy_values: list[float] = []
    for inftime in latencies:
        energy_values.append(avg_powerdraw * inftime)
    # update the record with energy/power
    avg_energy = float(np.mean(energy_values))  # inftimes are in seconds, mJ to J

    packet = (
        (avg_latency, avg_powerdraw, avg_energy),
        (avg_gpu_lat, avg_dla_lat),
        avg_gpu_preproc,
        avg_dla_preproc,
    )
    packet_str = f"(({packet[0][0]:.2f} ms, {packet[0][1]:.2f} W, {packet[0][2]:.2f} J), ({packet[1][0]:.2f} ms, {packet[1][1]:.2f} ms), {packet[2]:.2f} ms, {packet[3]:.2f} ms)"
    print(
        f"Benched: {pyolo.models[gpu_id].name} & {pyolo.models[dla_id].name} -> {packet_str}",
    )
    return packet


def conf2acc(conf: float, a: float = 4.0, b: float = 0.3, bias: float = 0.0) -> float:
    acc = 1 / (1 + math.exp(-a * (conf - b)))
    return min(acc + bias * (1 - acc), 1)


TRACE = True


class DRIFT:
    def __init__(
        self: Self,
        models: list[Path],
        priorities: dict[int, int],
        high_priority_latency: float = 33.0,
        high_priority_accuracy_goal: float = 0.5,
        low_priority_accuracy_goal: float = 0.4,
        energy_budget: float = 2.0,  # for now set very high, 10 J
        powerdraw_max: float = 60.0,  # for now set very high, 100 W
        time_per_frame: float = 66.0,
        highpri_pad: int = 128,
        annealing_min: float = 0.1,
        cache_path: Path = Path(__file__).parent / "cache.pickle",
        valdata_path: Path = Path(__file__).parent / "records" / "acc_f1_record.pickle",
        predictor_path: Path = Path(__file__).parent / "predictor.pickle",
    ) -> None:
        # assign the constraints
        self._hp_lat_limit = high_priority_latency
        self._lp_lat_limit = time_per_frame - high_priority_latency
        self._energy_budget = energy_budget
        self._powerdraw_max = powerdraw_max
        self._time_per_frame = time_per_frame
        self._a_min = annealing_min

        # create the predictor from the data source
        # for each model we have save the data into a name list and into a dict
        self._model_names: list[str] = [mpath.stem for mpath in models]
        self._predictor_data: dict[str, tuple[list[float], list[float]]] = {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        predictor_path.parent.mkdir(parents=True, exist_ok=True)
        self._predictor: AccuracyPredictor

        def _load_pred():
            with valdata_path.open("rb") as f:
                raw_data = pickle.load(f)
            for mname in self._model_names:
                self._predictor_data[mname] = raw_data[mname]
            if predictor_path.exists():
                with predictor_path.open("rb") as f:
                    self._predictor = pickle.load(f)
            else:
                self._predictor = AccuracyPredictor(self._model_names)
                self._predictor.fit(self._predictor_data)
                with predictor_path.open("wb+") as f:
                    pickle.dump(self._predictor, f, protocol=pickle.HIGHEST_PROTOCOL)

        _load_pred()
        try:
            assert self._model_names == self._predictor.model_names, (
                "Most likely using a saved predictor with different names"
            )
        except AssertionError:
            predictor_path.unlink()
            _load_pred()
        # add a small buffer for the prediction accuracys
        self._hp_pred_buffer: deque[list[float]] = deque(maxlen=10)
        self._lp_pred_buffer: deque[list[float]] = deque(maxlen=10)

        # define a scaling factor
        all_f1_data = []
        for _, f1_data in self._predictor_data.values():
            all_f1_data += f1_data
        median_f1 = float(np.median(all_f1_data))
        # assign the scaling factor such that an accuracy goal of 0.5 corresponding
        # to meeting the median F!
        self._acc_scale = 0.5 / median_f1

        # setup attributes for the heuristic
        self._priorities = priorities
        self._hp_accuracy_goal = high_priority_accuracy_goal / self._acc_scale
        self._lp_accuracy_goal = low_priority_accuracy_goal / self._acc_scale
        # print(self._hp_accuracy_goal, self._lp_accuracy_goal)

        # setup other attributes
        self._hp_pad = highpri_pad

        # partion models to be contained in ParallelYOLO
        self._pyolo = ParallelYOLO(
            engines=models,
            warmup_iterations=10,
            warmup=True,
        )
        assert self._pyolo._models is not None
        # create a shallow copy of the models, dont want to edit the underlying
        self._rmodels: list[YOLO] = self._pyolo.models
        # create GPU and DLA models lists
        self._gpu_idxs = []
        self._dla_idxs = []
        for idx, model in enumerate(self._rmodels):
            if "dla" in model.name:
                self._dla_idxs.append(idx)
            else:
                self._gpu_idxs.append(idx)

        # ensure that the rmodel names are the same as self._model_names and predictor
        assert [yolo.name for yolo in self._rmodels] == self._model_names
        assert [yolo.name for yolo in self._rmodels] == self._predictor.model_names

        # build the name to int and int to name map
        self._name_to_id: dict[str, int] = {
            yolo.name: idx for idx, yolo in enumerate(self._rmodels)
        }
        self._id_to_name: dict[int, str] = {
            idx: yolo.name for idx, yolo in enumerate(self._rmodels)
        }

        # need to assess the latency
        # benchmark each underlying engine and then save results
        self._cache: dict[
            str,
            tuple[tuple[float, float, float], tuple[float, float] | None, float, float],
        ]
        if not cache_path.exists():
            with cache_path.open("wb+") as f:
                pickle.dump({}, f, protocol=pickle.HIGHEST_PROTOCOL)
        with cache_path.open("rb") as f:
            self._cache = pickle.load(f)
        # benchmark each standalone engine
        for model in self._rmodels:
            if self._cache.get(model.name) is not None:
                continue
            infer, cpu_pre, cuda_pre = benchmark_yolo(model)
            self._cache[model.name] = (infer, None, cpu_pre, cuda_pre)
            # save the cache file now
            with cache_path.open("wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        # now we have to benchmark the parallel engines
        # assume only a GPU + DLA setup
        for gpu_id in self._gpu_idxs:
            for dla_id in self._dla_idxs:
                combo_name = self._get_name(gpu_id, dla_id)
                if self._cache.get(combo_name) is not None:
                    continue
                infer, indiv_inf, gpu_pre, dla_pre = benchmark_parallel_yolo(
                    self._pyolo, gpu_id, dla_id,
                )
                self._cache[combo_name] = (infer, indiv_inf, gpu_pre, dla_pre)
            # save the cache file now
            with cache_path.open("wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        # create a list of all possible inference setups for HP (this includes parallel combos)
        self._setups = []
        for gpu_id in self._gpu_idxs:
            # gpu + gpu assessment
            for gpu_id2 in self._gpu_idxs:
                # do want use same model, lose the no_copy optim, much slower
                if gpu_id == gpu_id2:
                    continue
                # assess if gpu + gpu is valid
                datapack1 = self._cache[self._get_name(gpu_id)]
                datapack2 = self._cache[self._get_name(gpu_id2)]
                (infer1, powerdraw1, energy1), _, _, cuda_pre1 = datapack1
                (infer2, powerdraw2, energy2), _, _, cuda_pre2 = datapack2
                if (
                    (infer1 + cuda_pre1 <= self._hp_lat_limit)
                    and (infer2 + cuda_pre2 <= self._lp_lat_limit)
                    and (energy1 + energy2 <= self._energy_budget)
                    and (powerdraw1 <= self._powerdraw_max)
                    and (powerdraw2 <= self._powerdraw_max)
                ):
                    self._setups.append(
                        ((gpu_id, gpu_id2), datapack1, datapack2),
                    )
            # gpu + dla assessment
            for dla_id in self._dla_idxs:
                combo_name = self._get_name(gpu_id, dla_id)
                datapack = self._cache[combo_name]
                (infer, powerdraw, energy), (gpu_lat, dla_lat), gpu_pre, dla_pre = (
                    datapack
                )
                if (
                    (infer <= self._time_per_frame)
                    and (energy <= self._energy_budget)
                    and (powerdraw <= self._powerdraw_max)
                    and (gpu_lat + gpu_pre <= self._hp_lat_limit)
                    and (dla_lat + dla_pre <= self._lp_lat_limit)
                ):
                    self._setups.append(((gpu_id, dla_id), datapack, None))

        if len(self._setups) == 0:
            err_msg = "Cannot make any valid schedules!"
            raise ValueError(err_msg)

        # sort the setups by lowest total time
        # need to add preproc and inference
        # for serial models add up the datapacks
        # for parallel use the overall infer time
        def total_infer(setup) -> float:
            _, datapack1, datapack2 = setup
            if datapack2 is None:
                # use the total inference time plus first model preprocessing
                (infer, _, _), (_, _), m1_preproc, _ = datapack1
                return infer + m1_preproc
            # otherwise we need to add up the m1 infer + preproc and m2 infer + preproc
            (infer1, _, _), _, _, m1_cuda_preproc = datapack1
            (infer2, _, _), _, _, m2_cuda_preproc = datapack2
            return m1_cuda_preproc + infer1 + m2_cuda_preproc + infer2

        self._setups = sorted(
            self._setups, key=lambda item: total_infer(item), reverse=False,
        )
        # sanity check of setup sizes
        for setup in self._setups:
            assert len(setup) == 3
        # by default use the setup which runs longest, also correspond to the most accurate
        # since V10M + V10M 1280 + 1120 will be that combination
        self._default = self._setups[-1]
        # hp_imgsz = self._rmodels[self._default[0][0]].input_shape[0]
        # lp_imgsz = self._rmodels[self._default[0][1]].input_shape[0]
        # for setup in self._setups:
        #     (m_id1, m_id2), _, _ = setup
        #     imgsz1 = self._rmodels[m_id1].input_shape[0]
        #     imgsz2 = self._rmodels[m_id2].input_shape[0]
        #     # if the new HP image size is larger, update
        #     # OR
        #     # if the HP image size is the same, but increase image size for LP
        #     if (imgsz1 > hp_imgsz) or (imgsz1 == hp_imgsz and imgsz2 > lp_imgsz):
        #         self._default = setup
        #         hp_imgsz = imgsz1
        #         lp_imgsz = imgsz2
        assert len(self._default) == 3
        # to start set the current setup to the default
        self._current = self._default
        self._use_baseline = False  # set the special case
        assert len(self._current) == 3

        # assign some extra static model information based on the setups
        # generate a list of all potential HP models we can use
        all_hp_mids = [s[0][0] for s in self._setups]
        self._potential_hp: list[int] = []
        for hp_mid in all_hp_mids:
            if hp_mid not in self._potential_hp:
                self._potential_hp.append(hp_mid)
        self._potential_hp_set = set(self._potential_hp)

        # state storage for previous detections
        self._lp_size = 0
        self._hp_bbox: tuple[int, int, int, int] | None = None
        self._prev_dets: tuple[Detections, Detections, Detections] = (
            [],
            [],
            [],
        )
        # store accuracy predictions
        self._pred_acc = 0.0
        # store the names of the models used
        self._inf_record: list[tuple[str, str]] = []

        # store the frame packing algorithm
        # self._packer = FramePacker((1920, 1080), gridsize=256)
        self._packer = AnnealingFramePacker(
            (1920, 1080), gridsize=1920 // 10, min_prob=self._a_min,
        )

        # store the sampling ratio, default model uses 1280 input size
        self._sample_ratio = 1280 / 1920

        # assign initial models
        self.reset()

        # # DEBUG INFO
        # # summarize all possible setups
        # def fmt_datapack(datapack) -> str:
        #     if datapack is None:
        #         return ""
        #     packet_str = "("
        #     for portion in datapack:
        #         if portion is None:
        #             packet_str += "None, "
        #             continue
        #         if isinstance(portion, tuple):
        #             packet_str += f"{tuple(map(lambda x: round(x, 2), portion))}, "
        #         else:
        #             packet_str += f"{round(portion, 2)}, "
        #     return f"{packet_str})"
        # for setup in self._setups:
        #     s_name = self._get_name(*setup[0])
        #     print(f"{s_name}: {fmt_datapack(setup[1])} & {fmt_datapack(setup[1])}")
        # print(f"Default: {self.get_current_models()}")

    def __del__(self: Self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            for model in self._rmodels:
                del model
        with contextlib.suppress(AttributeError, RuntimeError):
            for model in self._pyolo._models:
                model.__del__()
                del model
        with contextlib.suppress(AttributeError, RuntimeError):
            self._pyolo.__del__()
            del self._pyolo
        with contextlib.suppress(AttributeError, RuntimeError):
            del self._predictor

    def _get_name(self: Self, id1: int, id2: int | None = None) -> str:
        if id2 is not None:
            return f"{self._rmodels[id1].name}+{self._rmodels[id2].name}"
        return self._rmodels[id1].name

    def _get_latency(self, model: int | str) -> float:
        if isinstance(model, int):
            model = self._id_to_name[model]
        # print(model)
        infer, _, _, pre = self._cache[model]
        # infer is a tuple of inference time, powerdraw, energy
        return infer[0] + pre

    def _update_pred(self, model: str, conf: float, *, use_lp: bool) -> list[float]:
        # run the predictor, update the prediction buffer, average on buffer
        preds = self._predictor.predict(model, conf)

        # buffer
        buffer = self._lp_pred_buffer if use_lp else self._hp_pred_buffer
        buffer.append(preds)

        # average the lists into a new one
        def avg_data(buffer: deque[list[float]]) -> list[float]:
            avg_data: np.ndarray = np.mean(list(buffer), axis=0)
            avg_preds: list[float] = avg_data.tolist()
            return avg_preds

        return avg_data(buffer)

    @staticmethod
    def get_high_low_bboxes(
        bboxes: list[tuple[tuple[int, int, int, int], float, int]],
        priorities: dict[int, int],
    ) -> tuple[
        list[tuple[tuple[int, int, int, int], float, int]],
        list[tuple[tuple[int, int, int, int], float, int]],
    ]:
        high = []
        low = []
        for entry in bboxes:
            bbox, _, classid = entry
            area_thres = priorities[classid]

            # if low priority can stop right away
            if area_thres == -1:
                low.append(entry)
                continue

            # otherwise the entry has a priority
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area > area_thres:
                high.append(entry)
            else:
                low.append(entry)

        return high, low

    @staticmethod
    def get_bounding(
        bboxes: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> tuple[int, int, int, int]:
        return cv2ext.bboxes.bounding([bbox for bbox, _, _ in bboxes])

    def _run_single(self: Self, img: np.ndarray, id1: int) -> tuple[Output, None]:
        self._pyolo.submit_model(
            img, id1, preprocessed=False, postprocess=False, no_copy=True,
        )
        return (self._pyolo.retrieve_model(id1), None)

    def _run_sequential(
        self: Self, img1: np.ndarray, id1: int, img2: np.ndarray | None, id2: int,
    ) -> tuple[Output, Output | None]:
        # preprocess and run id1, then repeat for id2
        tensor1, ratio1, padding1 = self._pyolo.preprocess_model(
            img1, id1, resize="letterbox", no_copy=True,
        )
        self._pyolo.submit_model(
            tensor1, id1, preprocessed=True, postprocess=False, no_copy=True,
        )
        outputs1, _, _ = self._pyolo.retrieve_model(id1)
        output_pack2 = None
        if img2 is not None:
            self._pyolo.submit_model(
                img2, id2, preprocessed=False, postprocess=False, no_copy=True,
            )
            output_pack2 = self._pyolo.retrieve_model(id2)

        return (
            (outputs1, ratio1, padding1),
            output_pack2,
        )

    def _run_parallel(
        self: Self, img1: np.ndarray, id1: int, img2: np.ndarray | None, id2: int,
    ) -> tuple[Output, Output | None]:
        # preprocess id1 then submit
        # then preprocess and submit id2
        # wait for id1 then wait for id2
        tensor1, ratio1, padding1 = self._pyolo.preprocess_model(
            img1, id1, no_copy=True,
        )
        self._pyolo.submit_model(
            tensor1, id1, preprocessed=True, postprocess=False, no_copy=True,
        )
        output_pack2 = None
        if img2 is not None:
            tensor2, ratio2, padding2 = self._pyolo.preprocess_model(
                img2, id2, no_copy=True,
            )
            self._pyolo.submit_model(
                tensor2, id2, preprocessed=True, postprocess=False, no_copy=True,
            )

        outputs1, _, _ = self._pyolo.retrieve_model(id1)
        if img2 is not None:
            outputs2, _, _ = self._pyolo.retrieve_model(id2)
            output_pack2 = (outputs2, ratio2, padding2)

        return (
            (outputs1, ratio1, padding1),
            output_pack2,
        )

    def _run_pair(
        self: Self,
        img1: np.ndarray,
        id1: int,
        img2: np.ndarray,
        id2: int | None,
        *,
        parallel: bool | None = None,
    ) -> tuple[Output, Output | None]:
        if id2 is None:
            return self._run_single(img1, id1)
        if parallel:
            return self._run_parallel(img1, id1, img2, id2)
        return self._run_sequential(img1, id1, img2, id2)

    def _run_baseline(
        self: Self,
        img: np.ndarray,
    ) -> tuple[Output, None]:
        # just use the name_to_id mapping already existing
        model = self._rmodels[self._name_to_id["yolov10m_1280_fp16_gpu"]]
        tensor, ratio, padding = model.preprocess(img, no_copy=True)
        outputs = model.run(tensor, preprocessed=True, postprocess=False, no_copy=True)
        return (outputs, ratio, padding), None

    def get_hp_bbox(
        self: Self,
        bound_shape: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        if len(self._prev_dets[0]) == 0:
            return None
            # return (0, 0, bound_shape[0], bound_shape[1])
        # then we get the high priority boxes
        highpri_bbox = None
        high_dets, _ = DRIFT.get_high_low_bboxes(self._prev_dets[0], self._priorities)
        if len(high_dets) != 0:
            highpri_bbox = DRIFT.get_bounding(high_dets)
            x1, y1, x2, y2 = highpri_bbox
            # add some padding to the bbox
            y_bound, x_bound = float("inf"), float("inf")
            y_bound, x_bound = bound_shape
            x1 = int(max(0, x1 - self._hp_pad))
            y1 = int(max(0, y1 - self._hp_pad))
            x2 = int(min(x_bound, x2 + self._hp_pad))
            y2 = int(min(y_bound, y2 + self._hp_pad))
            highpri_bbox = (x1, y1, x2, y2)
        return highpri_bbox

    def get_hp_lp_images(
        self: Self, img: np.ndarray, hp_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        tuple[float, float],
        dict[tuple[int, int], tuple[int, int]],
    ]:
        height, width = img.shape[:2]
        if height != self._packer._height or width != self._packer._width:
            self._packer.reset((width, height), max(width, height) // 10)
            # also setup the correct sample ratio
            self._sample_ratio = 1280 / max(height, width)
        # second item is always the original image
        if hp_bbox is not None and not self._use_baseline:
            max_height, max_width = img.shape[:2]
            x1, y1, x2, y2 = hp_bbox
            # could make an optimization here extending the smaller dim to match the larger one
            # since input size is always square and always letterbox can grab extra "for free"
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width < bbox_height:
                diff = bbox_height - bbox_width
                x1 = int(max(0, x1 - diff // 2))
                x2 = int(min(max_width, x2 + diff // 2))
            elif bbox_height < bbox_width:
                diff = bbox_width - bbox_height
                y1 = int(max(0, y1 - diff // 2))
                y2 = int(min(max_height, y2 + diff // 2))
            # after the optim, will potentially have slightly more data in view
            cropped = img[y1:y2, x1:x2].copy()
            offset = (x1, y1)
            # pack the remainder
            lp_img, lp_map = self._packer.pack(img, hp_bbox)
            # assert lp_img.flags["C_CONTIGUOUS"]
            # if len(lp_map.keys()) == 0:
            #     lp_img = None
            # if lp_map.shape[0] * lp_map.shape[1] == 1:
            #     lp_img = None
            # assert cropped.flags["C_CONTIGUOUS"]
            self._lp_size = 0 if lp_img is None else lp_img.shape[:2]
            return cropped, lp_img, offset, lp_map
        if self._use_baseline:
            # return img, None, (0.0, 0.0), {}
            self._lp_size = 0
            return img, None, (0.0, 0.0), np.zeros((1, 1, 2))
        # return img, img, (0.0, 0.0), {}
        # print("Passing image twice")
        return img, img, (0.0, 0.0), np.zeros((1, 1, 2))

    def run(
        self: Self, hp_img: np.ndarray, lp_img: np.ndarray | None,
    ) -> tuple[Output, Output | None]:
        # if using the baseline
        if self._use_baseline:
            self._inf_record.append(("yolov10m_1280_fp16_gpu", None))
            return self._run_baseline(hp_img)
        (m_id1, m_id2), _, dpack2 = self._current
        parallel = False
        if dpack2 is None:
            parallel = True
        # store the inference record
        self._inf_record.append(
            (self._get_name(m_id1), self._get_name(m_id2) if m_id2 else "NA"),
        )

        # run the models
        return self._run_pair(hp_img, m_id1, lp_img, m_id2, parallel=parallel)

    def postprocess(
        self: Self,
        output1: Output,
        output2: Output | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
        output1, ratio1, padding1 = output1
        p_output1 = self._pyolo.postprocess_model(
            output1, self._current[0][0], ratio1, padding1, no_copy=True,
        )
        if output2 is not None:
            output2, ratio2, padding2 = output2
            p_output2 = self._pyolo.postprocess_model(
                output2, self._current[0][1], ratio2, padding2, no_copy=True,
            )
            return p_output1, p_output2
        return p_output1, None

    def get_detections(
        self: Self,
        output1: list[np.ndarray],
        output2: list[np.ndarray] | None,
        offset: tuple[float, float],
        lp_map: dict[tuple[int, int], tuple[int, int]],
    ) -> tuple[Detections, Detections, Detections]:
        dets1 = self._pyolo.get_detections_model(output1, self._current[0][0])

        dets2 = []
        if output2 is not None:
            dets2 = self._pyolo.get_detections_model(output2, self._current[0][1])
            dets2 = self._packer.unpack(dets2, lp_map)

        new_dets1 = []
        o_x, o_y = offset
        for bbox, conf, classid in dets1:
            x1, y1, x2, y2 = bbox
            x1 = int(x1 + o_x)
            y1 = int(y1 + o_y)
            x2 = int(x2 + o_x)
            y2 = int(y2 + o_y)
            new_dets1.append(((x1, y1, x2, y2), conf, classid))
        dets1 = new_dets1

        if len(dets2) == 0:
            all_dets = dets1
        else:
            all_dets = dets1 + dets2
            all_dets = cv2ext.bboxes.nms(all_dets, iou_threshold=0.5)

        # self._packer.update_dets(all_dets)
        self._packer.update(all_dets)

        self._prev_dets = (all_dets, dets1, dets2)

        return all_dets, dets1, dets2

    def rebalance(
        self: Self,
        hp_bbox: tuple[int, int, int, int] | None,
        lp_img: np.ndarray | None = None,
        *,
        pre: bool = False,
    ) -> tuple[int, int, int, int] | None:
        # ---------------------------------
        # Original frame size: X
        # Previous bounding box for HP objects: Y
        # Accuracy target for HP : A

        # m1 = lower parameter-size model with same accuracy as A
        # m2 = same-parameter-size model with resolution Y (accuracy is assumed to be A)
        # m3 = same-parameter-size model with resolution X

        # mTargetHP = m3

        # if m1 != NULL
        # if (latency(m1) < latency (m2))
        #     mTargetHP = m1
        # else if m2!=m3 // i.e., latency(m2) <latency (m3)
        #     mTargetHP = m2
        # else if m2 != m3
        # mTargetHP = m2

        # if mTargetHP == m3
        # ignore LP and fallback the baseline Yolo
        # else{
        # //Process LP model based on the LP_knob
        # latency_cap_for_LP = latency(mTargetHP)
        # mTargetLP-DLA = estimate DLA latency for LP, including overhead, by finding fastest DLA model satisfying LP_knob
        # mTargetLP-GPU = estimate GPU latency for LP , by finding fastest GPU model satisfying LP_knob

        # if mTargetLP-DLA < mTargetHP
        #     Execute LP on DLA in parallel with mTargetHP
        # else if mTargetLP-GPU+latency_cap_for_LP < latency(m3)
        #     Execute LP on GPU in series
        # else
        #     fallback to baseline Yolo
        # }
        # ---------------------------------

        used_baseline = self._use_baseline
        self._use_baseline = False
        # if we do not have previous detections do not change
        if len(self._prev_dets[0]) == 0:
            self._use_baseline = True
            return
        # if we cannot update the hp_bbox then do not change
        if hp_bbox is None:
            self._use_baseline = True
            return

        # get baseline informatiopn
        baseline_id = self._name_to_id["yolov10m_1280_fp16_gpu"]
        baseline_latency = self._get_latency("yolov10m_1280_fp16_gpu")

        # solve the hp area
        hx1, hy1, hx2, hy2 = hp_bbox
        hp_width, hp_height = hx2 - hx1, hy2 - hy1
        hp_max_dim = max(hp_width, hp_height)
        hp_area = (hx2 - hx1) * (hy2 - hy1)

        # get the predictions, if we used baseline manually set the name, otherwise auto
        hp_model = (
            self._id_to_name[self._current[0][0]]
            if not used_baseline
            else "yolov10m_1280_fp16_gpu"
        )
        hp_conf = (
            float(np.mean([conf for _, conf, _ in self._prev_dets[1]]))
            if len(self._prev_dets[1]) > 0
            else 0.0
        )
        hp_preds = self._update_pred(hp_model, hp_conf, use_lp=False)

        # NEW HEURSITIC - BBOX SIZE
        bbox_size = np.median(
            [
                ((x2 - x1) + (y2 - y1)) / 2.0
                for (x1, y1, x2, y2), _, _ in self._prev_dets[0]
            ],
        )
        img_w, img_h = self._packer._width, self._packer._height
        subsample_scale = 1.0 - (bbox_size / max(img_w, img_h))
        # print(subsample_scale)

        # get any which are greater-than-equal-to baseline
        # this would be the best candiate which would be faster and we estimate is at least as accurate
        faster_model = None
        faster_latency = baseline_latency
        baseline_acc_pred = hp_preds[baseline_id] * 1.0  # remove outliers
        # the base model will generate better dets than other models (in terms of confidence)
        # small scaling factor to account for this
        for idx, acc_pred in enumerate(hp_preds):
            if idx == baseline_id:
                continue
            # want to filter out DLA options, only want for secondary
            if "dla" in self._id_to_name[idx]:
                continue
            # # still only want to use medium models
            # if "10m" not in self._id_to_name[idx]:
            #     continue
            model = self._rmodels[idx]
            m_imgsz = model.input_shape[0]
            # m_area = m_imgsz ** 2
            # if m_area < hp_area:
            #     continue
            if m_imgsz / hp_max_dim < self._sample_ratio * subsample_scale:
                continue
            if acc_pred < baseline_acc_pred:
                continue
            new_latency = self._get_latency(idx)
            if new_latency < faster_latency:
                faster_model = idx
                faster_latency = new_latency
        # if faster_model is not None:
        #     print(f"{self._id_to_name[faster_model]} acc: {hp_preds[faster_model]}, latency: {faster_latency}")

        # find a version of YOLOv10M which has a smaller input size
        # which fits the HP bounding box
        smaller_model = None
        smaller_latency = baseline_latency
        for mid in range(len(self._rmodels)):
            if mid == baseline_id:
                continue
            model = self._rmodels[mid]
            if "10m" not in model.name:
                continue
            m_imgsz = model.input_shape[0]
            # if m_imgsz < hp_max_dim:
            #     continue
            # m_area = m_imgsz ** 2
            # if m_area < hp_area:
            #     continue
            if m_imgsz / hp_max_dim < self._sample_ratio * subsample_scale:
                continue
            new_latency = self._get_latency(mid)
            if new_latency < smaller_latency:
                smaller_model = mid
                smaller_latency = new_latency
        # if smaller_model is not None:
        #     print(f"{self._id_to_name[smaller_model]} acc: {hp_preds[smaller_model]}, latency: {smaller_latency}")

        # if we could not find a faster model, then use baseline
        if smaller_model is None and faster_model is None:
            self._use_baseline = True
            return

        # otherwise set the new hp_model_id to the faster of the two
        if faster_latency < smaller_latency:
            hp_model_id = faster_model
            hp_model_latency = faster_latency
        else:
            hp_model_id = smaller_model
            hp_model_latency = smaller_latency

        # ensure we did not pick the baseline
        if hp_model_id == baseline_id:
            self._use_baseline = True
            return

        # # here we can check if the LP region is None and "fast-forward" detection
        # if pre:
        #     if lp_img is None:
        #         self._current = (hp_model_id, None), None, None
        #         return

        # once we have a HP model id, second part of algorithm
        if not used_baseline:
            dets_to_estimate = self._prev_dets[2]
        else:
            dets_to_estimate = self._prev_dets[0]

        if self._current[0][1]:
            lp_model = self._id_to_name[self._current[0][1]]
        else:
            lp_model = self._id_to_name[self._current[0][0]]
        lp_conf = (
            float(np.mean([conf for _, conf, _ in dets_to_estimate]))
            if len(dets_to_estimate) > 0
            else 0.0
        )
        lp_preds = self._update_pred(lp_model, lp_conf, use_lp=True)

        # find all possible models which satisfy the latency constraint
        # solve for lp model which is above accuracy and minimizes latency
        second_model = None
        second_latency = -1
        # print("Schedule second model")
        for idx in range(len(lp_preds)):
            # do want to schedule baseline or model we picked
            if idx == baseline_id:
                continue
            if idx == hp_model_id:
                continue
            # do not meet accuracy, continue
            if lp_preds[idx] < self._lp_accuracy_goal:
                continue
            # get the latency of hp_model picked + this model
            # if the second is using dla get the combo setup
            if "dla" in self._id_to_name[idx]:
                # this gets the combined latency
                new_latency = self._get_latency(self._get_name(hp_model_id, idx))
            # otherwise we just get the normal latency since serial
            else:
                # add the latencies in serial
                new_latency = self._get_latency(idx) + hp_model_latency
            if new_latency > baseline_latency:
                continue
            # we found a schedule which meets accuracy and is faster than baseline
            if second_model is None or new_latency < second_latency:
                second_model = idx
                second_latency = new_latency
                # print(f"\t{self._id_to_name[idx]}, {new_latency}")

        # if we could not find a second model, then use baseline
        if second_model is None:
            self._use_baseline = True
            return

        # we found a new HP and LP model
        # get the corresponding schedule

        # the schedule can be made by setting the current value
        # current is a tuple of model ids, and two datapcks
        # the datapacks do nothing at runtime except, for datapack2
        # which is checked and is None will assume they run in parallel
        # so set based on whether second model name contains dla
        is_dla = "dla" in self._id_to_name[second_model]
        new_current = (hp_model_id, second_model), None, None if is_dla else True
        self._current = new_current
        self._use_baseline = False
        return

        # # OLD ALGORITHM

        # # # based on the previous detections, rebalance the low/high models
        # # # predict the current accuracy value
        # # def compute_acc(dets: Detections) -> float:
        # #     return conf2acc(float(np.mean([conf for (_, conf, _) in dets])))

        # # overall_acc = compute_acc(self._prev_dets[0])
        # # self._pred_acc = overall_acc
        # # # hp_acc = compute_acc(self._prev_dets[1])
        # # lp_acc = compute_acc(self._prev_dets[2])

        # # get the current model names
        # # get the previous hp and lp model names from self._current

        # pot_hp_model_ids: list[int] = [
        #     idx for idx, pred in enumerate(hp_preds) if pred >= self._hp_accuracy_goal
        # ]
        # # pot_lp_model_ids: list[int] = [
        # #     idx for idx, pred in enumerate(lp_preds) if pred >= self._lp_accuracy_goal
        # # ]
        # # this is the accuracy estimate of the current low priority model
        # lp_acc = lp_preds[self._current[0][1]]

        # # find models which are greater than the area
        # # we will use this to minimize latency
        # new_potential_hp = []
        # for mid in pot_hp_model_ids:
        #     model = self._rmodels[mid]
        #     m_imgsz = model.input_shape[0]
        #     m_area = m_imgsz ** 2
        #     if m_area >= hp_area and mid in self._potential_hp_set:
        #         new_potential_hp.append(mid)

        # # if no such models exist, then we maximize F1
        # # since the self._setups is sorted ascending, thus self._potential_hp
        # # is sorted ascending, so we pick the HP as simply grabbin the last one
        # new_hp_mid = self._potential_hp[hp_preds.index(max(hp_preds))]
        # # if we do have a small HP area, minimize latency
        # if len(new_potential_hp) > 0:
        #     new_hp_mid = new_potential_hp[0]

        # # now get the possible setups using this HP model id
        # # still latency sorted for HP, instead latency sort for total infer time
        # new_setups = [setup for setup in self._setups if setup[0][0] == new_hp_mid]

        # # # compute the current low priority image size
        # # m_ids, _, _ = self._current
        # # c_lp_imgsz = self._rmodels[m_ids[1]].input_shape[0]
        # # # compute the new lowest acceptable image size
        # # t_lp_imgsz = c_lp_imgsz
        # # t_acc = lp_acc  # store a temp copy to modify on
        # # if lp_acc < self._lp_accuracy_goal:
        # #     while t_acc < self._lp_accuracy_goal:
        # #         if t_acc + 0.1 < self._lp_accuracy_goal:
        # #             t_lp_imgsz += 160  # step up the image size
        # #             t_acc += 0.1
        # #             continue
        # #         # the diff is too small to make a change on
        # #         break
        # # else:
        # #     while t_acc > self._lp_accuracy_goal:
        # #         if t_acc - 0.1 > self._lp_accuracy_goal:
        # #             t_lp_imgsz -= 160
        # #             t_acc -= 0.1
        # #             continue
        # #         # the diff is too small to make a change on
        # #         break

        # # filter the setups which have a second model meeting the target imgsz
        # filtered_new_setups = []
        # for setup in new_setups:
        #     _, m_id2 = setup[0]
        #     model2 = self._rmodels[m_id2]
        #     # also need to check if we meet the accuracy method
        #     if lp_preds[m_id2] >= self._lp_accuracy_goal:
        #     #     continue
        #     # m_imgsz2 = model2.input_shape[0]
        #     # if m_imgsz2 >= t_lp_imgsz:
        #         filtered_new_setups.append(setup)

        # # if there are no valid setups, we need to spend more to meet accuracy
        # # so use the slowest
        # new_current = new_setups[-1]

        # # since we never unsort the values, simply optimize for overall inference time
        # # ALTERNATIVE:
        # # however, we should be optimizing for (image_size) / (added infernce time)
        # # finds the most efficient (computationally speaking)
        # # secondary model
        # if len(filtered_new_setups) > 0:
        #     new_current = filtered_new_setups[0]

        # # assign the current models to the new model ids
        # self._current = new_current

    def reset(self: Self) -> None:
        # clear detections
        self._prev_dets = ([], None, None)
        # store accuracy predictions
        self._pred_acc = 0.0
        # clear record of inferences
        self._inf_record = []

        # by default use the slowest models
        # presumably will be most accurate
        self._current = self._default
        self._use_baseline = False

        # reset the frame packer
        self._packer.reset((1920, 1080), 1920 // 10)

        # reset sample ratio
        self._sample_ratio = 1280 / 1920

    def get_current_models(self: Self) -> str:
        (m_id1, m_id2), _, _ = self._current
        return self._get_name(m_id1, m_id2)


def main() -> None:
    """Test the dynamic method."""
    from pathlib import Path

    from cv2ext import Display, VideoWriter, bboxes
    from cv2ext.image import draw

    from drift import mot_reader
    from drift.paths import YOLO_DLA0_PATHS, YOLO_GPU_PATHS
    from drift.priorities import PEOPLE

    # video storage info
    vid_dir = Path(__file__).parent / "videos"

    # engine_dir = Path(__file__).parent.parent.parent / "data" / "tensorrt_engines" / "orinagx" / "jetpack602"
    # yolo = engine_dir / "yolov10" / "yolov10m_1280_fp16_gpu.engine", version=10, warmup=True)
    dynamic_yolo_models = [mpath for (mname, mpath) in YOLO_GPU_PATHS if "10" in mname]
    for mname, mpath in YOLO_DLA0_PATHS:
        if "xs" in mname and ("480" in mname or "320" in mname):
            dynamic_yolo_models.append(mpath)
        # for mname, mpath in YOLO_DLA1_PATHS:
        if "xm" in mname and ("480" in mname or "640" in mname):
            dynamic_yolo_models.append(mpath)

    dynamic = DRIFT(
        dynamic_yolo_models,
        PEOPLE,
        low_priority_accuracy_goal=0.5,
        highpri_pad=64,
    )

    inference = []
    stiching = []
    scheduling = []

    reader: mot_reader.MotReader
    with Display("display") as display:
        for mvers in [17]:
            for reader in mot_reader.MotRunner(
                mot_version=mvers,
                detection=True,
            ):
                # if "MOT17-13" not in reader.name:
                #     continue

                image: np.ndarray
                gt_label: list[tuple[tuple[int, int, int, int], int]]

                dynamic.reset()

                with VideoWriter(vid_dir / f"{reader.name}.mp4", fps=25) as writer:
                    for i, (image, _, gt_label) in enumerate(reader):
                        height, width = image.shape[:2]
                        if display.stopped:
                            break

                        # drawing info
                        gt_bboxes = [bbox for (bbox, _) in gt_label]
                        canvas = image

                        # actual inference
                        t0 = time.time()
                        h_bbox = dynamic.get_hp_bbox(image.shape[:2])
                        hp_img, lp_img, hp_offset, lp_map = dynamic.get_hp_lp_images(
                            image, h_bbox,
                        )
                        t1 = time.time()
                        dynamic.rebalance(h_bbox, lp_img)
                        t2 = time.time()
                        hp_outputs, lp_outputs = dynamic.run(hp_img, lp_img)
                        hp_p_outputs, lp_p_outputs = dynamic.postprocess(
                            hp_outputs, lp_outputs,
                        )
                        all_dets, hp_dets, lp_dets = dynamic.get_detections(
                            hp_p_outputs, lp_p_outputs, hp_offset, lp_map,
                        )
                        t3 = time.time()

                        # grab the model ids
                        (m_id1, m_id2), _, _ = dynamic._current
                        used_baseline = dynamic._use_baseline
                        baseline_id = dynamic._name_to_id["yolov10m_1280_fp16_gpu"]

                        # # actual inference
                        # t0 = time.time()
                        # h_bbox = dynamic.get_hp_bbox(image.shape[:2])
                        # hp_img, lp_img, hp_offset, lp_map = dynamic.get_hp_lp_images(image, h_bbox)
                        # t1 = time.time()
                        # hp_outputs, lp_outputs = dynamic.run(hp_img, lp_img)
                        # hp_p_outputs, lp_p_outputs = dynamic.postprocess(hp_outputs, lp_outputs)
                        # all_dets, hp_dets, lp_dets = dynamic.get_detections(hp_p_outputs, lp_p_outputs, hp_offset, lp_map)
                        # t2 = time.time()

                        # # grab the model ids
                        # (m_id1, m_id2), _, _ = dynamic._current
                        # used_baseline = dynamic._use_baseline
                        # baseline_id = dynamic._name_to_id["yolov10m_1280_fp16_gpu"]

                        # t3 = time.time()
                        # dynamic.rebalance(h_bbox)
                        # t4 = time.time()

                        inference.append((t3 - t2) * 1000.0)
                        stiching.append((t1 - t0) * 1000.0)
                        scheduling.append((t2 - t1) * 1000.0)

                        if lp_img is not None and i % 50 == 0:
                            frame_dir = Path(__file__).parent / "frames" / reader.name
                            frame_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(frame_dir / f"{i}.png"), lp_img)

                        # setup profiling dump
                        def ts(t: tuple[float, float]) -> float:
                            return round((t[1] - t[0]) * 1000.0, 2)

                        def modelinfo(id: int) -> tuple[float, float, float, str]:
                            model = dynamic._rmodels[id]
                            preproc = model._pre_profile
                            infer = model._infer_profile
                            post = model._post_profile
                            return (
                                ts(preproc),
                                ts(infer),
                                ts(post),
                                "DLA" if "dla" in model.name else "GPU",
                            )

                        # print(f"FRAME: {i}")
                        # if used_baseline:
                        #     print(f"BASELINE: {modelinfo(baseline_id)}")
                        # else:
                        #     print(f"MODEL1: {modelinfo(m_id1)}")
                        #     print(f"MODEL2: {modelinfo(m_id2)}")

                        # # convert the detections to the motrack dets
                        # dets = [
                        #     PredBBox(Point(x1, y1), Point(x2, y2), classid, conf) for (x1, y1, x2, y2), conf, classid in all_dets
                        # ]
                        # tracklets = tracker.track(tracklets, dets, i)
                        # active_tracklets = [t for t in tracklets if t.state == TrackletState.ACTIVE]
                        # tracked_boxes = []
                        # ids = []
                        # for t in active_tracklets:
                        #     tracked_boxes.append(
                        #         nxyxy_to_xyxy(
                        #             (
                        #                 t.bbox.upper_left.x,
                        #                 t.bbox.upper_left.y,
                        #                 t.bbox.bottom_right.x,
                        #                 t.bbox.bottom_right.y,
                        #             ),
                        #             width,
                        #             height,
                        #         ),
                        #     )
                        #     ids.append(t.id)

                        # # frame_entry = []
                        # # for bboxid, bbox in zip(ids, bboxes):
                        # #     # bbox is left, top, width, height
                        # #     x, y, w, h = xyxy_to_xywh(bbox)
                        # #     # use j + 1 since frame id is 1 indexed
                        # #     entry = (j + 1, bboxid, x, y, w, h)
                        # #     frame_entry.append(entry)
                        # # motwriter.add(frame_entry)

                        # draw GT
                        bboxes.draw_bboxes(canvas, gt_bboxes, color=Color.RED)
                        # draw low bboxes or all bboxes
                        if lp_dets is not None:
                            low_bboxes = [bbox for (bbox, _, _) in lp_dets]
                            bboxes.draw_bboxes(canvas, low_bboxes, color=Color.BLUE)
                        # if all_dets is not None:
                        #     all_bboxes = [bbox for (bbox, _, _) in all_dets]
                        #     bboxes.draw_bboxes(canvas, all_bboxes, color=Color.BLUE)
                        # draw high bboxes
                        if hp_dets is not None:
                            high_bboxes = [bbox for (bbox, _, _) in hp_dets]
                            bboxes.draw_bboxes(canvas, high_bboxes, color=Color.GREEN)
                        if h_bbox is not None:
                            draw.rectangle(canvas, h_bbox, color=Color.GREEN)

                        # # draw the tracked bounding boxes
                        # bboxes.draw_bboxes(canvas, tracked_boxes, color=Color.WHITE)

                        # curr_models = dynamic.get_current_models()
                        # hp, lp = curr_models.split("+")

                        # draw.text(canvas, hp, (10, 50))
                        # draw.text(canvas, lp, (10, 90))

                        # # display frame and write result
                        display(canvas)
                        # writer.write(
                        #     cv2.resize(
                        #         canvas,
                        #         dsize=None,
                        #         fx=0.66,
                        #         fy=0.66,
                        #         interpolation=cv2.INTER_LINEAR,
                        #     ),
                        # )
                        # time.sleep(0.001)

    inf_mean = float(np.mean(inference))
    stich_mean = float(np.mean(stiching))
    schedule_mean = float(np.mean(scheduling))
    total_time = inf_mean + stich_mean + schedule_mean
    print(f"Inf mean: {inf_mean}")
    print(f"Inf % time: {inf_mean / total_time}")
    print(f"Stich mean: {stich_mean}")
    print(f"Stich % time: {stich_mean / total_time}")
    print(f"Schedule: {schedule_mean}")
    print(f"Schedule %: {schedule_mean / total_time}")

    del dynamic


if __name__ == "__main__":
    set_log_level("INFO")
    main()
