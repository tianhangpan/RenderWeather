from pathlib import Path
import argparse
import threading
import multiprocessing
import json
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import scipy
import shutil
import h5py
from PIL import Image
import cv2

from weather_renderer import WeatherRenderer


class DatasetProcessor:
    def __init__(self):
        self.num_workers = multiprocessing.cpu_count()
        self.dataset_dir = Path(r'/Users/pantianhang/python_data/datasets/nwpu')
        stages = ['train', 'val', 'test']
        self.img_dirs = set({})
        for stage in stages:
            stage_img_dirs = set((self.dataset_dir / stage / 'images').glob('*.jpg'))
            self.img_dirs |= stage_img_dirs
        self.img_dirs = list(self.img_dirs)

        self.threads = [threading.Thread(target=self.task, args=(i,)) for i in range(self.num_workers)]
        self.present_index = 0
        self.done_number = 0
        self.lock = threading.RLock()

    def task(self, processor_id):
        pass

    def render_weather(self):
        pass

    def render_gt_density_map(self):
        pass
