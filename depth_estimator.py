import numpy as np
import torch
from PIL import Image
import os
import time
from pathlib import Path
import cv2


class DepthEstimator:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.manual_seed(int(time.time()))
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.device = torch.device('cuda')

        repo = "isl-org/ZoeDepth"
        self.model_cpu = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
        self.model_gpu = self.model_cpu.to(self.device)

        self.dataset_dir = Path(r'E:\python_data\datasets\nwpu')
        self.stage_dirs = [(self.dataset_dir / stage) for stage in ['train', 'val', 'test']]
        self.img_dirs = set({})
        for e in self.stage_dirs:
            stage_img_dirs = set((e / 'images').glob('*.jpg'))
            self.img_dirs |= stage_img_dirs

            stage_depth_dir = e / 'depth'
            stage_depth_dir.mkdir(exist_ok=True)

        self.processed = set({})
        for e in self.stage_dirs:
            depth_dir = e / 'depth'
            depth_dir.mkdir(exist_ok=True)
            stage_depth_dirs = list(depth_dir.glob('*.npz'))
            for i in range(len(stage_depth_dirs)):
                tmp = stage_depth_dirs[i]
                tmp = Path(str(tmp).replace('depth', 'images').replace('npz', 'jpg'))
                stage_depth_dirs[i] = tmp
            self.processed |= set(stage_depth_dirs)

        self.img_dirs -= self.processed
        self.img_dirs = list(self.img_dirs)
        print(f'the length of task list: {len(self.img_dirs)}')

    def process(self):
        for i, d in enumerate(self.img_dirs):
            image = Image.open(d)
            try:
                depth = self.model_gpu.infer_pil(image, output_type="tensor")
            except:
                depth = self.model_cpu.infer_pil(image, output_type="tensor")
            depth = np.array(depth)
            depth = self.guided_filter(depth, depth, 32)
            np.savez_compressed(str(d).replace('images', 'depth').replace('.jpg', '.npz'), depth=depth)
            print(f'[{i} / {len(self.img_dirs)}] done. ')

    @staticmethod
    def guided_filter(img, p, win_size):
        # borrowed from https://blog.csdn.net/wsp_1138886114/article/details/84228939
        eps = 0.01

        mean_i = cv2.blur(img, win_size)  # I的均值平滑
        mean_p = cv2.blur(p, win_size)  # p的均值平滑

        mean_ii = cv2.blur(img * img, win_size)  # I*I的均值平滑
        mean_ip = cv2.blur(img * p, win_size)  # I*p的均值平滑

        var_i = mean_ii - mean_i * mean_i  # 方差
        cov_ip = mean_ip - mean_i * mean_p  # 协方差

        a = cov_ip / (var_i + eps)  # 相关因子a
        b = mean_p - a * mean_i  # 相关因子b

        mean_a = cv2.blur(a, win_size)  # 对a进行均值平滑
        mean_b = cv2.blur(b, win_size)  # 对b进行均值平滑

        q = mean_a * img + mean_b
        return q


if __name__ == '__main__':
    depth_estimator = DepthEstimator()
    depth_estimator.process()
