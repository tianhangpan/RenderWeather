from pathlib import Path
import shutil


class Collator:
    def __init__(self):
        self.dataset_dir = Path(r'/Users/pantianhang/python_data/datasets/nwpu')
        stages = ['images_part1', 'images_part2', 'images_part3', 'images_part4', 'images_part5a', 'images_part5b']
        self.img_dirs = set({})
        for stage in stages:
            stage_img_dirs = set((self.dataset_dir / stage).glob('*.jpg'))
            self.img_dirs |= stage_img_dirs
        self.img_dirs = list(self.img_dirs)

        self.target_dir = Path(r'/Users/pantianhang/python_data/datasets/nwpu_collated')
        self.target_dir.mkdir(exist_ok=True)
        stages = ['train', 'val', 'test']
        for stage in stages:
            tar_stage_dir = self.target_dir / stage
            tar_stage_dir.mkdir(exist_ok=True)
            tar_stage_img_dir = tar_stage_dir / 'images'
            tar_stage_img_dir.mkdir(exist_ok=True)

    def process(self):
        independent_files = ['readme.md', 'test.txt', 'train.txt', 'val.txt']
        for name in independent_files:
            source_dir = self.dataset_dir / name
            shutil.copy(source_dir, self.target_dir)

        for i, img_dir in enumerate(self.img_dirs):
            num = int(img_dir.stem)
            stage = None
            if 1 <= num <= 3109:
                stage = 'train'
            elif 3110 <= num <= 3609:
                stage = 'val'
            elif 3610 <= num <= 5109:
                stage = 'test'
            img_tar_dir = self.target_dir / stage / 'images'
            shutil.copy(img_dir, img_tar_dir)
            print(f'[{i} / {len(self.img_dirs)}] done. ')


if __name__ == '__main__':
    collator = Collator()
    collator.process()

