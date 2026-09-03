import json
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Iterator
import threading
from queue import Queue
import time


class COCO17ValLoader:
    def __init__(self, base_path: str, buffer_size: int = 16):
        """
        Initialize the COCO17 validation dataset loader.
        
        Args:
            base_path: Path to the directory containing 'annotations' and 'val2017' folders
            buffer_size: Size of the buffer queue for preloading images
        """
        self.images_path = Path(base_path) / "val"
        self.annotations_file = Path(base_path) / 'annotations' / 'instances_val2017.json'
        self.buffer_size = buffer_size
        self.buffer = Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        
        # Load and process annotations
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
            
        # Create image_id to annotations mapping
        self.annotations_map = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_map:
                self.annotations_map[img_id] = []
            
            # Convert bbox from xywh to xyxy format
            x, y, w, h = ann['bbox']
            bbox = (int(x), int(y), int(x + w), int(y + h))
            self.annotations_map[img_id].append((bbox, ann['category_id'] - 1))
        
        # Create list of image files with their IDs
        self.images = []
        for img in data['images']:
            self.images.append((img['file_name'], img['id']))
        
        # Initialize threading related attributes
        self.current_idx = 0
        self.thread = None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def _load_images(self):
        """Background thread function for loading images into buffer"""
        while not self.stop_event.is_set() and self.current_idx < len(self.images):
            if self.buffer.full():
                time.sleep(0.01)  # Sleep briefly if buffer is full
                continue
                
            img_file, img_id = self.images[self.current_idx]
            img_path = self.images_path / img_file
            img = cv2.imread(str(img_path))
            annotations = self.annotations_map.get(img_id, [])
            
            self.buffer.put((img, annotations))
            self.current_idx += 1
            
        # Add None to signal the end of the dataset
        if not self.stop_event.is_set():
            self.buffer.put(None)
    
    def __iter__(self) -> Iterator[tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]]:
        """
        Iterate over the dataset, yielding (image, annotations) pairs.
        
        Returns:
            Iterator yielding tuples of:
                - image: numpy array (H, W, C)
                - annotations: list of (bbox, class_id) tuples where bbox is (x1, y1, x2, y2)
        """
        # Reset state
        self.current_idx = 0
        self.stop_event.clear()
        
        # Clear any remaining items in buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except:
                break
        
        # Start loader thread
        self.thread = threading.Thread(target=self._load_images, daemon=True)
        self.thread.start()
        
        # Yield items from buffer
        while True:
            item = self.buffer.get()
            if item is None:  # End of dataset
                break
            yield item
            
        # Cleanup
        self.stop_event.set()
        if self.thread:
            self.thread.join()
    
    def __del__(self):
        """Cleanup when the object is deleted"""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
