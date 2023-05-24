import numpy as np
import torch
from PIL import Image
import os
import time
from pathlib import Path


class DepthEstimator:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.manual_seed(int(time.time()))
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.device = torch.device('cuda')

        repo = "isl-org/ZoeDepth"
        self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
        self.model = self.model.to(self.device)

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
            stage_depth_dirs = list(depth_dir.glob('*.jpg'))
            for i in range(len(stage_depth_dirs)):
                tmp = stage_depth_dirs[i]
                tmp = Path(str(tmp).replace('depth', 'images'))
                stage_depth_dirs[i] = tmp
            self.processed |= set(stage_depth_dirs)

        self.img_dirs -= self.processed
        self.img_dirs = list(self.img_dirs)
        print(f'the length of task list: {len(self.img_dirs)}')

    def process(self):
        for i, d in enumerate(self.img_dirs):
            image = Image.open(d)
            depth = self.model.infer_pil(image, output_type="tensor")
            depth = np.array(depth)
            np.save(str(d).replace('images', 'depth').replace('.jpg', '.npy'), depth)
            # depth.save(str(d).replace('images', 'depth'))
            print(f'[{i} / {len(self.img_dirs)}] done. ')


if __name__ == '__main__':
    depth_estimator = DepthEstimator()
    depth_estimator.process()
