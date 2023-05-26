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
import random

from weather_renderer import WeatherRenderer


class DatasetProcessor:
    def __init__(self):
        self.num_workers = multiprocessing.cpu_count()
        args = self.parse_args()
        self.dataset_dir = Path(args.dataset_dir)
        assert args.task in ['weather', 'gt_density']
        self.task = args.task
        stages = ['train', 'val', 'test']
        self.img_dirs = set({})
        for stage in stages:
            stage_img_dirs = set((self.dataset_dir / stage / 'images').glob('*.jpg'))
            self.img_dirs |= stage_img_dirs
        self.img_dirs = list(self.img_dirs)

        if args.task == 'weather':
            for stage in stages:
                stage_weather_dir = self.dataset_dir / stage / 'weather_images'
                stage_weather_dir.mkdir(exist_ok=True)
                stage_weather_json_dir = self.dataset_dir / stage / 'weather_jsons'
                stage_weather_json_dir.mkdir(exist_ok=True)
        else:
            for stage in stages:
                stage_gt_dir = self.dataset_dir / stage / 'gt_density_maps'
                stage_gt_dir.mkdir(exist_ok=True)

        self.threads = [threading.Thread(target=self.target, args=(i,)) for i in range(self.num_workers)]
        self.present_index = 0
        self.done_number = 0
        self.lock = threading.RLock()
        print(f'count of threads: {self.num_workers}')

    def start(self):
        for thread in self.threads:
            thread.daemon = True
            thread.start()

        for thread in self.threads:
            thread.join()

    def target(self, processor_id):
        while True:
            self.lock.acquire()
            if self.present_index >= len(self.img_dirs):
                self.lock.release()
                return
            img_dir = self.img_dirs[self.present_index]
            self.present_index += 1
            self.lock.release()

            if self.task == 'weather':
                self.render_weather(img_dir)
            else:
                self.render_gt_density_map(img_dir)

            self.lock.acquire()
            self.done_number += 1
            print(f'Thread {processor_id:<2}: [{self.done_number}/{len(self.img_dirs)}] '
                  f'{img_dir.name} processed. ')
            self.lock.release()

    @staticmethod
    def render_weather(img_path):
        rand = random.random()
        weather = (2 if rand <= .05 else 1) if rand <= .1 else 0  # 0 for normal, 1 for haze and 2 for rain
        img_target_path = Path(str(img_path).replace('images', 'weather_images'))

        if weather == 0:
            shutil.copy(img_path, img_target_path)
        elif weather == 1:
            img = np.array(Image.open(img_path))
            depth_path = str(img_path).replace('images', 'depth').replace('.jpg', '.npz')
            depth = np.load(depth_path)['depth']
            img_haze, _, _ = WeatherRenderer.render_haze(img, depth)
            Image.fromarray((img_haze * 255).astype(np.uint8)).save(img_target_path, quality=95)
        else:
            img = np.array(Image.open(img_path))
            img_rain, _ = WeatherRenderer.render_rain(img, 'random', 0.15, 1.2)
            Image.fromarray((img_rain * 255).astype(np.uint8)).save(img_target_path, quality=95)

        json_path = Path(str(img_path).replace('images', 'jsons').replace('.jpg', '.json'))
        with json_path.open('r') as jsf:
            state_dict = json.load(jsf)
        weather_json_path = Path(str(json_path).replace('jsons', 'weather_jsons'))
        with weather_json_path.open('w') as jsf:
            dict_s = json.dumps(state_dict)
            jsf.write(dict_s)

    @staticmethod
    def render_gt_density_map(img_dir):
        img = np.array(Image.open(img_dir))
        with Path(str(img_dir).replace('images', 'jsons').replace('.jpg', '.json')).open('r') as jsf:
            state_dict = json.load(jsf)
            points = state_dict['points']

        gt_density_map = DatasetProcessor.gaussian_filter_density(img, points)
        gt_save_path = Path(str(img_dir).replace('images', 'gt_density_maps').replace('.jpg', '.npz'))
        np.save(gt_save_path, gt_density_map=gt_density_map)

    @staticmethod
    def gaussian_filter_density(img, points):
        """
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

        return:
        density: the density-map we want. Same shape as input image but only has one channel.

        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        img_shape: (768,1024) 768 is row and 1024 is column.
        """
        img_shape = [img.shape[0], img.shape[1]]
        # print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        leafsize = 2048
        # build kdtree
        tree = KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=4)

        # print('generate density...')
        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            else:
                sigma = np.average(np.array(pt.shape)) / 2. / 2.  # case: 1 point
            sigma = min(20., sigma)
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        # print('done.')
        density = np.clip(density, 0., 1.)
        return density

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        # parser.add_argument('dataset_dir', type=str, help='original data path')
        # parser.add_argument('task', type=str, help='target data path')

        args = parser.parse_args()
        args.dataset_dir = r'/Users/pantianhang/python_data/datasets/nwpu'
        args.task = 'weather'
        return args
