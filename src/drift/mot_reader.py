# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: BLE001, S301, DOC501, S403
"""Read the MOT dataset labels."""

from __future__ import annotations

import configparser
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from cv2ext import IterableVideo
from tqdm import tqdm


@lru_cache(maxsize=None)
def get_mot_labels(
        mot_dir: Path | str | None = None,
        mot_version: int | None = None,
        *,
        detection: bool | None = None,
        test: bool | None = None,
    ) -> list[tuple[Path, Path | None, Path | None, Path, Path]]:
    """
    Get the list of MOT label files
    
    Parameters
    ----------
    mot_dir : Path | str | None
        The directory to search for the mot17/20 directories
        If None will use: Path(__file__).parent
        The directory structure should be:
        > mot_dir:
        ----> mot17
        ---------> test
        ---------> train
        ----> mot20
        ---------> test
        ---------> train
    mot_version : int
        The version of the MOT dataset to get labels for, by
        default None.
        Options are 17, 20.
    detection : bool, optional
        Whether or not to use MOTDet labels.
    test : bool, optional
        Whether to get the test set, by default False.

    Returns
    -------
    list[tuple[Path, Path | None, Path | None, Path, Path]]
        seqinfo.ini, det.txt, gt.txt, video.avi, subdirectory
    
    """
    if mot_version not in [None, 17, 20]:
        err_msg = "Invalid MOT version. Options are 17, 20. "
        err_msg += f"Provided: {mot_version}"
        raise ValueError(err_msg)
    if mot_dir is None:
        mot_dir = Path(__file__).parent
    if isinstance(mot_dir, str):
        mot_dir = Path(mot_dir)
    subfolder = "test" if test else "train"
    if mot_version is not None:
        mot_subdir = f"mot{mot_version}det" if detection else f"mot{mot_version}"
        mot_directories = [mot_dir / mot_subdir / subfolder]
    else:
        mot_subdirs = [
            "mot17det" if detection else "mot17",
            "mot20det" if detection else "mot20",
        ]
        mot_directories = [
            Path(__file__).parent / mot_subdir / subfolder
            for mot_subdir in mot_subdirs
        ]
    mot_labels = []
    
    for directory in mot_directories:
        subdirs = sorted(directory.iterdir())
        for subdirectory in subdirs:
            det_label = subdirectory / "det" / "det.txt"
            if not det_label.exists():
                det_label = None
            gt_label = subdirectory / "gt" / "gt.txt"
            if not gt_label.exists():
                gt_label = None
            seqinfo = subdirectory / "seqinfo.ini"
            video = None
            # for vidpath in ["video.mp4", "video.avi"]:
            for vidpath in ["video.mp4", "video.avi", "video_resized.mp4", "video_resized.avi"]:
                vp = subdirectory / vidpath
                vp = vp.resolve()
                # print(vp)
                if vp.exists():
                    video = vp
                    break
            # print(f"Found video: {video}")
            if video is None:
                err_msg = f"Could not find valid video file in dir: {subdirectory}"
                raise FileNotFoundError(err_msg)
            mot_labels.append((seqinfo, det_label, gt_label, video, subdirectory))
    
    return mot_labels


def read_mot_label(
    label: Path,
    seqlen: int,
) -> list[list[tuple[tuple[int, int, int, int], int]]]:
    """
    Read the ground truth MOT label file.
    
    Parameters
    ----------
    label : Path
        The path to the ground truth label file.
    seqlen : int
        The path to the sequence info file.
        
    Returns
    -------
    list[list[tuple[tuple[int, int, int, int], int]]]
        The list of ground truth labels.
        Each label is a list of bounding boxes.

    """
    data: list[list[tuple[tuple[int, int, int, int], int]]] = []
    for _ in range(seqlen):
        data.append([])

    with label.open("r") as f:
        lines = f.readlines()

    # each object ID has frames
    for idx, line in enumerate(lines):
        values = line.strip().split(",")
        if len(values) == 10:
            fid, oid, x, y, w, h, conf, _, _, _ = values
        elif len(values) == 9:
            fid, oid, x, y, w, h, conf, _, _ = values
        elif len(values) == 8:
            fid, oid, x, y, w, h, conf, _ = values
        elif len(values) == 7:
            fid, oid, x, y, w, h, conf = values
        # elif len(values) == 6:
        #     fid, oid, x, y, w, h = values
        else:
            err_msg = f"Invalid number of entries in {label.name} found on line: {idx + 1}"
            raise ValueError(err_msg)
        if int(conf) == 0:
            # print("SKipping!!!!")
            continue
        fid = int(fid)
        oid = int(oid)
        x, y, w, h = map(float, (x, y, w, h))
        data[fid - 1].append(
            ((int(x), int(y), int(x + w), int(y + h)), oid)
            )

    return data


class MotReader:
    """Helper class for reading mot labels and video frames."""

    def __init__(
        self,
        video_id: int = 0,
        mot_version: int | None = None,
        *,
        detection: bool | None = None,
        test: bool | None = None,
        resize: bool | None = None,
    ) -> None:
        """
        Initialize the MotReader object.
        
        Parameters
        ----------
        video_id : int
            The video id to load from the MOT dataset.
            Does not have a 1:1 correspondence, instead loads
            the index of found videos.
        mot_version : int, optional
            The version of the MOT dataset to get labels for, by
            default None.
            Options are 17, 20.
        detection : bool, optional
            Whether or not to use the 'det' versions of the MOT datasets.
            By default False.
        test : bool, optional
            Whether to get the test set, by default False.
        resize : bool, optional
            Whether or not to resize images to original size during loading.
            By default True

        """
        self._data = get_mot_labels(mot_version=mot_version, test=test, detection=detection)
        self._video_id = video_id
        self._label_idx = 0
        self._resize = resize
        self._curr_label = self._data[self._video_id]
        self._video = IterableVideo(self._curr_label[3])
        self._video_size = len(self._video)
        # try det labels if they exist
        self._det_labels = None
        if self._curr_label[1] is not None:
            self._det_labels = read_mot_label(self._curr_label[1], self._video_size)
        # try gt labels if they exist
        self._gt_labels = None
        if self._curr_label[2] is not None:
            self._gt_labels = read_mot_label(self._curr_label[2], self._video_size)
        # read the seqinfo file to get frame size
        config = configparser.ConfigParser()
        config.read(self._curr_label[0])
        self._frame_size = (int(config["Sequence"]["imWidth"]), int(config["Sequence"]["imHeight"]))

    @property
    def name(self) -> str:
        """Get the name of the MOT video."""

        return self._data[self._video_id][-1].stem
    
    @property
    def framesize(self) -> tuple[int, int]:
        """Get the (width, height) of each frame."""

        return self._frame_size
    
    def __iter__(self) -> MotReader:
        """Return the iterator object."""
        return self
    
    def __next__(self) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]] | None, list[tuple[tuple[int, int, int, int], int]] | None]:
        """Return the next frame and labels."""
        # will handle the StopIteration call for this object
        _, frame = next(self._video)

        det_label = None
        if self._det_labels is not None:
            det_label = self._det_labels[self._label_idx]
        gt_label = None
        if self._gt_labels is not None:
            gt_label = self._gt_labels[self._label_idx]
        self._label_idx += 1

        # resize the frame to original size
        # all videos are saved in 640, 480 but some images
        # are supposed to be 1920, 1080
        if self._resize:
            frame = cv2.resize(frame, self._frame_size)

        return frame, det_label, gt_label
    
    def __len__(self) -> int:
        """Return the number of labels."""
        return self._video_size


class MotRunner:
    """Wrapper for MotReader, allowing iterating over videos."""

    def __init__(
        self,
        mot_version: int | None = None,
        *,
        detection: bool | None = None,
        test: bool | None = None,
        resize: bool | None = None,
    ) -> None:
        """
        Create an instance of MotRunner
        
        Parameters
        ----------
        mot_version : int, optional
            Which MOT version to read data from.
            Options are 17, 20, or None (17 is default)
        detection : bool, optional
            Whether or not to load the detection sequences
        test : bool, optional
            Whether or not to use the test set
        resize : bool, optional
            Whether or not to resize images during loading
        
        """
        self._mot_version = mot_version
        self._test = test
        self._detection = detection
        self._resize = None if resize is None else resize
        self._numvids = len(get_mot_labels(mot_version=self._mot_version, test=self._test, detection=self._detection))
        self._current_vid = 0
        self._reader = None

    @property
    def reader(self) -> MotReader:
        """Get the current MotReader."""
        return self._reader

    def __iter__(self) -> MotRunner:
        """Return the iterator object."""
        return self

    def __next__(self) -> MotReader:
        """Get the next MotReader instance."""
        if self._current_vid >= self._numvids:
            raise StopIteration
    
        self._reader = MotReader(
            video_id=self._current_vid,
            mot_version=self._mot_version,
            test=self._test,
            detection=self._detection,
            resize=self._resize,
            )
        self._current_vid += 1
        return self._reader
    
    def __len__(self) -> int:
        """Get the number of MOT videos."""
        return self._numvids


if __name__ == "__main__":
    for reader in MotRunner(mot_version=17):
        for frame_info in tqdm(reader):
            pass
