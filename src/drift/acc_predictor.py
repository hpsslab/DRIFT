from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cv2ext.bboxes import iou as compute_iou
from trtutils.impls.yolo import YOLO

from drift import coco_reader
from drift.paths import DEFAULT_ENGINE_ROOT, discover_engines

# IMPORTS FOR THE MODEL PREDICTOR

from sklearn.linear_model import LinearRegression

# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.svm import SVR


def generate_f1_3d_topo_plots(
    model_data: dict[str, tuple[list[float], list[float]]],
    output_dir: Path,
    model_name: str,
    figsize: tuple = (8, 8),
    cmap: str = "plasma",
) -> None:
    """
    Generate a 3D surface plot for F1 scores.

    Args:
        model_data (dict): Dictionary with model names as keys and tuples as values.
                           The second item in the tuple is the list of F1 scores.
        output_dir (Path): Directory where the plots will be saved.
        model_name (str): The name of the model being plotted.
        figsize (tuple): Size of the plot.
        cmap (str): Color map for the surface plot.

    """
    import pandas as pd

    # Map param sizes to numeric values for 3D plotting
    param_size_map = {"t": 2, "n": 2, "s": 1, "m": 0}
    found_tiny = False
    found_nano = False

    plot_data = []

    for model, (_, f1_scores) in model_data.items():
        # Extract configuration from the model name
        config = model.split("_")
        psize = ""
        input_size = config[1]
        for param in ["t", "n", "s", "m"]:
            if param in config[0]:
                psize = param.replace("yolov", "").replace("yolox", "")
                if param == "t":
                    found_tiny = True
                if param == "n":
                    found_nano = True
                break
        f1_types = set()
        f1_types.update(type(f1) for f1 in f1_scores)
        plot_data.append(
            {
                "Param Size": param_size_map[psize],
                "Input Size": int(input_size),
                "F1": float(np.mean(f1_scores)),
            },
        )

    if found_nano and not found_tiny:
        param_labels = ["nano", "small", "medium"]
    elif not found_nano and found_tiny:
        param_labels = ["tiny", "small", "medium"]
    else:
        param_labels = ["nano", "tiny", "small", "medium"]
    param_labels.reverse()

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    # Create unique sorted lists of param sizes and input sizes
    param_sizes = sorted(df["Param Size"].unique())
    input_sizes = sorted(df["Input Size"].unique(), reverse=False)

    print(param_sizes, input_sizes)

    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(input_sizes, param_sizes)
    Z = np.zeros_like(X, dtype=float)

    # Compute the global min and max for normalization
    all_metric_values = df["F1"].values
    metric_min, metric_max = all_metric_values.min(), all_metric_values.max()

    # Fill Z matrix with normalized F1 scores
    for i, param in enumerate(param_sizes):
        for j, input_size in enumerate(input_sizes):
            mask = (df["Param Size"] == param) & (df["Input Size"] == input_size)
            if not df[mask].empty:
                raw_value = df[mask]["F1"].iloc[0]
                Z[i, j] = (raw_value - metric_min) / (
                    metric_max - metric_min
                )  # Normalize

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.8)

    # Customize plot
    ax.set_title(
        f"{model_name.upper().replace('V', 'v')}",
        fontsize=24,
        fontweight="bold",
        pad=0,
        y=0.96,
    )
    ax.set_xlabel("Input Size", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Parameter Size", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_zlabel("Normalized F1 Score", fontsize=18, fontweight="bold", labelpad=10)

    # Set y-axis ticks to parameter size labels
    ax.set_yticks(range(len(param_labels)))
    ax.set_yticklabels(param_labels, fontsize=16)
    ax.set_xticks(input_sizes)
    ax.set_xticklabels(input_sizes, fontsize=16, rotation=0)
    zstep = [i / 5 for i in range(0, 6, 1)]
    ax.set_zticks(zstep)
    ax.set_zticklabels(zstep, fontsize=16)

    # Adjust view angle for better visualization
    ax.view_init(elev=25, azim=145)

    # Save plot
    output_path = output_dir / f"{model_name}_3d_topo.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.close()

    # Trim the fat off the images
    import cv2

    plot_img = cv2.imread(output_path)
    height, width = plot_img.shape[:2]
    trimmed = plot_img[400 : height - 100, 150 : width - 50]
    cv2.imwrite(output_path, trimmed)


class SimpleRegressor:
    def __init__(self):
        # Using MultiOutputRegressor to handle multi-output regression
        # self.model = xgboost.XGBRegressor(objective='reg:squarederror')
        self.model = LinearRegression()
        # self.model = MultiOutputRegressor(LinearRegression())

    def fit(self, features: list[float], labels: list[list[float]]):
        """
        Train the model on the given data.
        :param features: List of single float features.
        :param labels: List of lists (multi-output target values).
        """
        X = np.array(features).reshape(-1, 1)  # Reshape for sklearn
        Y = np.array(labels)
        self.model.fit(X, Y)

    def predict(self, features: float) -> list[float]:
        """
        Predict labels for the given features.
        :param features: List of single float features.
        :return: Predicted list of lists (multi-output).
        """
        X = np.array([features]).reshape(-1, 1)  # Reshape for sklearn
        return self.model.predict(X).tolist()[0]


class AccuracyPredictor:
    def __init__(self, names: list[str]):
        self._models: dict[str, SimpleRegressor] = {}
        self._model_names = names  # Maintain order of model names

    def fit(self, data: dict[str, tuple[list[float], list[float]]]):
        """
        Fit a MultiOutputRegressor using data from all classes.
        :param data: Dictionary where keys are class names and values are tuples of (features, labels).
        """
        # maintain a keyed dictionary of LinearRegression models
        # the feature is a single float, the label is a list[float]

        train_data: dict[str, tuple[list[float], list[list[float]]]] = {}

        for name in self._model_names:
            # for each conf value, generate the corresponding f1 values
            conf_data = data[name][0]
            f1_data: list[list[float]] = []
            for i in range(len(conf_data)):
                # the features we want to predict are the same indices
                f1s = [data[n][1][i] for n in self._model_names]
                f1_data.append(f1s)

            train_data[name] = (conf_data, f1_data)

        # create the regressors
        for name, (features, labels) in train_data.items():
            print(f"Training regressor for: {name}")
            # print(features)
            regressor = SimpleRegressor()
            regressor.fit(features, labels)
            self._models[name] = regressor
            print("\tDone")

    def predict(self, model: str, conf: float) -> list[float]:
        """
        Predict labels for all classes using the input features.
        :param features: List of features to predict.
        :return: List of predictions in the order of model_names.
        """
        return self._models[model].predict(conf)

    @property
    def model_names(self) -> list[str]:
        """
        Return the list of model names in the order of predictions.
        """
        return self._model_names


def match_boxes(
    pred_boxes: list[
        tuple[tuple[int, int, int, int], float, int]
    ],  # bbox, conf, class_id
    gt_boxes: list[tuple[tuple[int, int, int, int], int]],  # bbox, class_id
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> list[tuple[int, int]]:
    """
    Match predicted boxes to ground truth boxes based on IoU and class IDs.

    Args:
        pred_boxes: List of (x1, y1, x2, y2, confidence, class_id)
        gt_boxes: List of ((x1, y1, x2, y2), class_id)
        iou_threshold: Minimum IoU for a match
        class_agnostic: If True, ignore class IDs when matching

    Returns:
        List of (pred_idx, gt_idx) matched pairs

    """
    matches = []
    used_gt = set()

    # Sort predictions by confidence (highest first)
    pred_indices = sorted(
        range(len(pred_boxes)), key=lambda i: pred_boxes[i][1], reverse=True,
    )

    for pred_idx in pred_indices:
        pred_box, conf, pred_class = pred_boxes[pred_idx]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_class) in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            # Skip if classes don't match (unless class_agnostic)
            if not class_agnostic and pred_class != gt_class:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)

    return matches


def calculate_metrics(
    pred_boxes: list[
        tuple[tuple[int, int, int, int], float, int]
    ],  # bbox, conf, class_id
    gt_boxes: list[tuple[tuple[int, int, int, int], int]],  # bbox, class_id
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, float]:
    """
    Calculate detection metrics including class-aware matching.

    Args:
        pred_boxes: List of (x1, y1, x2, y2, confidence, class_id)
        gt_boxes: List of ((x1, y1, x2, y2), class_id)
        iou_threshold: Minimum IoU for a match
        class_agnostic: If True, ignore class IDs when matching

    Returns:
        Dictionary of metrics including per-class stats

    """
    # if predictions are non and gt is None, perfect!
    if not pred_boxes and not gt_boxes:
        return {
            "overall": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,  # Perfect precision by convention
                "recall": 1.0,  # Perfect recall by convention
                "f1": 1.0,  # Perfect F1 by convention
                "accuracy": 1.0,  # Perfect accuracy
            },
            "per_class": {},
        }

    matches = match_boxes(pred_boxes, gt_boxes, iou_threshold, class_agnostic)

    # Calculate overall metrics
    true_positives = len(matches)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else int(len(gt_boxes) == len(pred_boxes))
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else int(len(gt_boxes) == len(pred_boxes))
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else int(len(gt_boxes) == len(pred_boxes))
    )
    accuracy = (
        true_positives / len(gt_boxes)
        if len(gt_boxes) != 0
        else int(len(gt_boxes) == len(pred_boxes))
    )

    # Calculate per-class metrics
    class_stats = {}
    if not class_agnostic:
        # Get unique classes
        pred_classes = set(box[2] for box in pred_boxes)
        gt_classes = set(box[1] for box in gt_boxes)
        all_classes = pred_classes.union(gt_classes)

        for class_id in all_classes:
            # Filter boxes for this class
            class_pred_boxes = [box for box in pred_boxes if box[2] == class_id]
            class_gt_boxes = [box for box in gt_boxes if box[1] == class_id]

            # Calculate metrics for this class
            class_matches = match_boxes(
                class_pred_boxes, class_gt_boxes, iou_threshold, True,
            )
            class_tp = len(class_matches)
            class_fp = len(class_pred_boxes) - class_tp
            class_fn = len(class_gt_boxes) - class_tp

            class_precision = (
                class_tp / (class_tp + class_fp) if class_tp + class_fp > 0 else 1e-6
            )
            class_recall = (
                class_tp / (class_tp + class_fn) if class_tp + class_fn > 0 else 1e-6
            )
            class_f1 = (
                2 * (class_precision * class_recall) / (class_precision + class_recall)
                if class_precision + class_recall > 0
                else 1e-6
            )

            class_stats[class_id] = {
                "tp": max(class_tp, 0),
                "fp": max(class_fp, 0),
                "fn": max(class_fn, 0),
                "precision": max(class_precision, 1e-6),
                "recall": max(class_recall, 1e-6),
                "f1": max(class_f1, 1e-6),
            }

    return {
        "overall": {
            "tp": max(true_positives, 0),
            "fp": max(false_positives, 0),
            "fn": max(false_negatives, 0),
            "precision": max(precision, 1e-6),
            "recall": max(recall, 1e-6),
            "f1": max(f1_score, 1e-6),
            "accuracy": max(accuracy, 1e-6),
        },
        "per_class": class_stats,
    }


def get_yolo(yolo_path: Path) -> YOLO:
    # check if there is yolox in path and sets scaling accordingly
    if "yolox" in yolo_path.stem:
        return YOLO(yolo_path, warmup=True, input_range=(0, 255))
    return YOLO(yolo_path, warmup=True)


def get_data(
    record_dir: Path | None = None,
    coco_dir: Path | None = None,
    engine_root: Path = DEFAULT_ENGINE_ROOT,
):
    import pickle

    from tqdm import tqdm

    if record_dir is None:
        record_dir = Path(__file__).parent / "records"
    if coco_dir is None:
        coco_dir = Path(__file__).parent.parent.parent / "datasets" / "coco17"
    record_dir.mkdir(parents=True, exist_ok=True)
    f1_record_file = record_dir / "acc_f1_record.pickle"
    precision_record_file = record_dir / "acc_precision_record.pickle"
    recall_record_file = record_dir / "acc_recall_record.pickle"

    # make the records
    f1_record: dict[str, tuple[list[float], list[float]]] = {}
    precision_record: dict[str, tuple[list[float], list[float]]] = {}
    recall_record: dict[str, tuple[list[float], list[float]]] = {}

    # f1 record
    if f1_record_file.exists():
        with f1_record_file.open("rb") as f:
            f1_record = pickle.load(f)
    else:
        with f1_record_file.open("wb+") as f:
            pickle.dump(f1_record, f, protocol=pickle.HIGHEST_PROTOCOL)
    # precision record
    if precision_record_file.exists():
        with precision_record_file.open("rb") as f:
            precision_record = pickle.load(f)
    else:
        with precision_record_file.open("wb+") as f:
            pickle.dump(precision_record, f, protocol=pickle.HIGHEST_PROTOCOL)
    # recall record
    if recall_record_file.exists():
        with recall_record_file.open("rb") as f:
            recall_record = pickle.load(f)
    else:
        with recall_record_file.open("wb+") as f:
            pickle.dump(recall_record, f, protocol=pickle.HIGHEST_PROTOCOL)

    coco = coco_reader.COCO17ValLoader(
        str(coco_dir),
    )

    # for model_paths in :
    #     models = {
    #         model.stem: get_yolo(model) for _, model in model_paths if "yolov10" in model.stem or "yolox" in model.stem
    #     }

    # models = {
    #         model.stem: get_yolo(model) for _, model in YOLO_DLA1_PATHS if "yolov10" in model.stem or "yolox" in model.stem
    #     }
    gpu_paths, dla0_paths, dla1_paths, _ = discover_engines(engine_root)
    for _, model_path in [*gpu_paths, *dla0_paths, *dla1_paths]:
        if not ("yolov10" in model_path.stem or "yolox" in model_path.stem):
            continue
        yolo = get_yolo(model_path)
        if f1_record.get(yolo.name) is not None:
            continue

        print(f"Processing: {yolo.name}")
        # mean conf values (per frame)
        # f1 values (per frame)
        confs: list[float] = []
        f1s: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []

        for img, labels in tqdm(coco):
            preds = yolo.end2end(img)
            mean_conf = float(np.mean([conf for _, conf, _ in preds])) if preds else 0.0
            confs.append(mean_conf)
            metrics = calculate_metrics(preds, labels)
            f1s.append(metrics["overall"]["f1"])
            precisions.append(metrics["overall"]["precision"])
            recalls.append(metrics["overall"]["recall"])

        # save the conf, f1 data
        f1_record[yolo.name] = (confs, f1s)
        precision_record[yolo.name] = (confs, precisions)
        recall_record[yolo.name] = (confs, recalls)

        for record_path, record in [
            (f1_record_file, f1_record),
            (precision_record_file, precision_record),
            (recall_record_file, recall_record),
        ]:
            with record_path.open("wb") as f:
                pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

        del yolo


def train(record_dir: Path | None = None):
    import pickle

    if record_dir is None:
        record_dir = Path(__file__).parent / "records"
    f1_record_file = record_dir / "acc_f1_record.pickle"
    recall_record_file = record_dir / "acc_recall_record.pickle"

    metrics = ["f1", "recall"]
    for idx, record_file in enumerate([f1_record_file, recall_record_file]):
        with record_file.open("rb") as f:
            record = pickle.load(f)

        # create a simple model
        model = AccuracyPredictor(record.keys())
        model.fit(record)

        results = validate_accuracy(model, record)
        print(metrics[idx])
        print(results)


def validate_accuracy(
    model: AccuracyPredictor,
    validation_data: dict[str, tuple[list[float], list[float]]],
    error_type="mae",
) -> dict[str, float]:
    """
    Validate the model's accuracy on the provided dataset using either MAE or MSE.
    :param model: The trained model (AccuracyPredictor instance).
    :param validation_data: A dictionary containing validation data (features, labels).
    :param error_type: 'mae' for Mean Absolute Error or 'mse' for Mean Squared Error.
    :return: A dictionary with error scores for each model.
    """
    from tqdm import tqdm

    accuracy_results = {}

    for model_name in tqdm(model.model_names):
        model_errors = []

        features, true_labels = validation_data[model_name]

        # Get predictions for the validation data
        for feature, true_label in zip(features, true_labels):
            predicted_label = model.predict(model_name, feature)

            # Calculate error based on chosen metric
            if error_type == "mae":
                # Mean Absolute Error
                error = np.mean(
                    np.abs(np.array(predicted_label) - np.array(true_label)),
                )
            elif error_type == "mse":
                # Mean Squared Error
                error = np.mean((np.array(predicted_label) - np.array(true_label)) ** 2)
            else:
                raise ValueError("Invalid error type. Use 'mae' or 'mse'.")

            model_errors.append(error)

        # Calculate average error for the model
        accuracy_results[model_name] = np.mean(model_errors)

    return accuracy_results


if __name__ == "__main__":
    import pickle

    get_data()
    record_dir = Path(__file__).parent / "records"

    for metric in ["f1", "precision", "recall"]:
        record_file = record_dir / f"acc_{metric}_record.pickle"
        if not record_file.exists():
            get_data()

        # generate the plot
        with record_file.open("rb") as f:
            data = pickle.load(f)
        for key in list(data.keys()):
            if "10" not in key:
                del data[key]
        generate_f1_3d_topo_plots(
            data, Path(__file__).parent / "plots", f"YOLOv10 {metric}",
        )

    # train()
