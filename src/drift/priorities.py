from __future__ import annotations

from collections import defaultdict

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

# car, motorcycle, bus, truck
VEHICLES = defaultdict(lambda: -1)
VEHICLES[2] = 0
VEHICLES[7] = 0

# people
PEOPLE = defaultdict(lambda: -1)
PEOPLE[0] = 0

# all
ALL = defaultdict(lambda: -1)
ALL[0] = 0
ALL[2] = 0
ALL[7] = 0
