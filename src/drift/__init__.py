# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""DRIFT: dynamic scheduling of multi-DNN inference on heterogeneous accelerators."""

from __future__ import annotations

from drift.acc_predictor import AccuracyPredictor
from drift.rewrite import DRIFT

__all__ = ["DRIFT", "AccuracyPredictor"]
__version__ = "0.1.0"
