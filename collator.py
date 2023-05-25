from pathlib import Path
import shutil
from PIL import Image
import json


class Collator:
    def __init__(self):
        self.dataset_dir = Path(r'/Users/pantianhang/Downloads/NWPU-Crowd')
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
            tar_stage_jsons_dir = tar_stage_dir / 'jsons'
            tar_stage_jsons_dir.mkdir(exist_ok=True)

    def process(self):
        independent_files = ['readme.md', 'test.txt', 'train.txt', 'val.txt']
        for name in independent_files:
            source_dir = self.dataset_dir / name
            shutil.copy(source_dir, self.target_dir)

        for i, img_dir in enumerate(self.img_dirs):
            num = int(img_dir.stem)
            if 1 <= num <= 2609:
                stage = 'train'
            elif 3110 <= num <= 3609:
                stage = 'val'
            elif 2610 <= num <= 3109:
                stage = 'test'
            else:
                continue

            img_tar_dir = self.target_dir / stage / 'images' / img_dir.name
            img, factor = self.resize(Image.open(img_dir))
            img = img.convert('RGB')
            img.save(img_tar_dir, quality=95)

            json_dir = self.dataset_dir / 'jsons' / (img_dir.stem + '.json')
            with json_dir.open('r') as jsf:
                state_dict = json.load(jsf)

            for j in range(len(state_dict['points'])):
                state_dict['points'][j][0] *= factor
                state_dict['points'][j][1] *= factor

            json_tar_dir = self.target_dir / stage / 'jsons' / json_dir.name
            with json_tar_dir.open('w') as jsf:
                dict_s = json.dumps(state_dict)
                jsf.write(dict_s)

            print(f'[{i} / {len(self.img_dirs)}] done. ')

    @staticmethod
    def resize(img: Image):
        w, h = img.width, img.height
        factor = 1
        if max(w, h) > 2048:
            factor = 2048 / max(w, h)
        elif min(w, h) < 512:
            factor = 512 / min(w, h)
            if max(w, h) * factor > 2048:
                factor = 2048 / max(w, h)
        return img.resize((round(w * factor), round(h * factor)), Image.BILINEAR), factor


if __name__ == '__main__':
    collator = Collator()
    collator.process()
