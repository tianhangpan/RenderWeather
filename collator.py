from pathlib import Path
import shutil
from PIL import Image
import json
import scipy
import random


class BaseCollator:
    def __init__(self, dataset_dir, target_dir, stages):
        self.dataset_dir = dataset_dir
        self.img_dirs = set({})
        for stage in stages:
            stage_img_dirs = set((self.dataset_dir / stage).glob('*.jpg'))
            self.img_dirs |= stage_img_dirs
        self.img_dirs = list(self.img_dirs)

        self.target_dir = target_dir
        self.target_dir.mkdir(exist_ok=True)
        stages = ['train', 'val', 'test']
        for stage in stages:
            tar_stage_dir = self.target_dir / stage
            tar_stage_dir.mkdir(exist_ok=True)
            tar_stage_img_dir = tar_stage_dir / 'images'
            tar_stage_img_dir.mkdir(exist_ok=True)
            tar_stage_jsons_dir = tar_stage_dir / 'jsons'
            tar_stage_jsons_dir.mkdir(exist_ok=True)

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


class NWPUCollator(BaseCollator):
    def __init__(self):
        super().__init__(Path(r'/Users/pantianhang/Downloads/NWPU-Crowd'), 
                         Path(r'/Users/pantianhang/python_data/datasets/nwpu_collated'), 
                         ['images_part1', 'images_part2', 'images_part3', 
                          'images_part4', 'images_part5a', 'images_part5b'])

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

            print(f'[{i + 1} / {len(self.img_dirs)}] done. ')


class UCFCollator(BaseCollator):
    def __init__(self):
        super().__init__(Path(r'/Users/pantianhang/python_data/datasets/UCF-QNRF_ECCV18'),
                         Path(r'/Users/pantianhang/python_data/datasets/UCF-QNRF_collated'),
                         ['Train', 'Test'])

    def process(self):
        for i, img_dir in enumerate(self.img_dirs):
            raw_stage = img_dir.parent.name
            raw_num = int(img_dir.name[4: 8])
            stage = None
            num = None
            if raw_stage == 'Train' and 1 <= raw_num <= (1201 - 334):
                stage = 'train'
                num = raw_num
            elif raw_stage == 'Train' and (1201 - 334 + 1) <= raw_num <= 1201:
                stage = 'val'
                num = raw_num
            elif raw_stage == 'Test':
                stage = 'test'
                num = raw_num + 1201
            num = f'{num:04d}'

            img_tar_dir = self.target_dir / stage / 'images' / f'{num}.jpg'
            img, factor = self.resize(Image.open(img_dir))
            img = img.convert('RGB')
            img.save(img_tar_dir, quality=95)

            mat_dir = Path(str(img_dir).replace('.jpg', '_ann.mat'))
            mat_file = scipy.io.loadmat(str(mat_dir))
            points = mat_file['annPoints']
            points = (points * factor).to_list()
            human_num = len(points)
            state_dict = {'human_num': human_num, 'points': points}
            json_tar_dir = self.target_dir / stage / 'jsons' / f'{num}.json'
            with json_tar_dir.open('w') as jsf:
                dict_s = json.dumps(state_dict)
                jsf.write(dict_s)

            print(f'[{i + 1} / {len(self.img_dirs)}] done. ')


class JHUCollator:
    def __init__(self):
        self.dataset_dir = Path(r'/Users/pantianhang/python_data/datasets/jhu_crowd_v2.0')
        self.target_dir = Path(r'/Users/pantianhang/python_data/datasets/jhu_collated')
        self.target_dir.mkdir(exist_ok=True)
        self.img_dirs = set({})
        for stage in ['train', 'val', 'test']:
            (self.target_dir / stage).mkdir(exist_ok=True)
            (self.target_dir / stage / 'images').mkdir(exist_ok=True)
            (self.target_dir / stage / 'jsons').mkdir(exist_ok=True)
            stage_img_dirs = set((self.dataset_dir / stage / 'images').glob('*.jpg'))
            self.img_dirs |= stage_img_dirs
        self.img_dirs = list(self.img_dirs)

    def process(self):
        weather_labels = {'train': {}, 'val': {}, 'test': {}}
        for stage in weather_labels.keys():
            with (self.dataset_dir / stage / 'image_labels.txt').open('r') as txtf:
                lines = list(txtf.readlines())
            for i in range(len(lines)):
                lines[i] = lines[i].split(',')
                weather_labels[stage][lines[i][0]] = int(lines[i][3])

        for i, img_dir in enumerate(self.img_dirs):
            stage = img_dir.parent.parent.name
            img_tar_dir = self.target_dir / stage / 'images' / img_dir.name
            img, factor = BaseCollator.resize(Image.open(img_dir))
            img = img.convert('RGB')
            img.save(img_tar_dir, quality=95)

            state_dict = {'image_id': img_dir.name}
            json_tar_dir = self.target_dir / stage / 'jsons' / f'{img_dir.stem}.json'
            txt_dir = Path(str(img_dir).replace('images', 'gt').replace('.jpg', '.txt'))
            with txt_dir.open('r') as txtf:
                lines = list(txtf.readlines())
            for j in range(len(lines)):
                lines[j] = list(map(int, (lines[j].split())[:2]))
                lines[j][0] *= factor
                lines[j][1] *= factor
            state_dict['points'] = lines
            state_dict['human_num'] = len(lines)
            state_dict['weather'] = weather_labels[stage][img_dir.stem]

            with json_tar_dir.open('w') as jsf:
                dict_s = json.dumps(state_dict)
                jsf.write(dict_s)

            print(f'[{i + 1} / {len(self.img_dirs)}] done. ')


class GCCCollator:
    def __init__(self):
        self.dataset_dir = Path(r'D:\python_data\datasets\GCC_raw')
        self.target_dir = Path(r'D:\python_data\datasets\GCC_collated')
        self.target_dir.mkdir(exist_ok=True)

        self.scene_lists = {'train': [], 'val': [], 'test': []}

        lists = {'train': self.dataset_dir / 'cross_location_train_list.txt',
                 'test': self.dataset_dir / 'cross_location_test_list.txt'}
        for stage in ['train', 'test']:
            with lists[stage].open('r') as txtf:
                for line in txtf:
                    data = line.split()
                    scene_number = data[3][11: 13]
                    if scene_number not in self.scene_lists[stage]:
                        self.scene_lists[stage].append(scene_number)

        self.scene_lists['val'] = sorted(random.sample(self.scene_lists['train'], 10))
        self.scene_lists['train'] = sorted(list(set(self.scene_lists['train']) - set(self.scene_lists['val'])))

        self.image_path_sets = {'train': set({}), 'val': set({}), 'test': set({})}
        for stage in ['train', 'val', 'test']:
            (self.target_dir / stage).mkdir(exist_ok=True)
            (self.target_dir / stage / 'images').mkdir(exist_ok=True)
            (self.target_dir / stage / 'jsons').mkdir(exist_ok=True)

            for scene_number in self.scene_lists[stage]:
                for camera_number in ['0', '1', '2', '3']:
                    sub_dir = self.dataset_dir / f'scene_{scene_number}_{camera_number}' / 'pngs'
                    self.image_path_sets[stage] |= set(sub_dir.glob('*.png'))

            self.image_path_sets[stage] = sorted(list(self.image_path_sets[stage]))

    def process(self):
        weather_dict = {'CLEAR': 0, 'CLOUDS': 1, 'RAIN': 2, 'FOGGY': 3, 'THUNDER': 4, 'OVERCAST': 5,
                        'EXTRASUNNY': 6}
        for stage in ['train', 'val', 'test']:
            for i, img_path in enumerate(self.image_path_sets[stage]):
                img = Image.open(img_path)
                img.save(self.target_dir / stage / 'images' / f'{img_path.stem}.jpg')

                json_path = Path(str(img_path).replace('pngs', 'jsons').replace('.png', '.json'))
                with json_path.open('r') as jsf:
                    state_dict = json.load(jsf)
                    points = state_dict['image_info']
                    corrected_points = [[point[1], point[0]] for point in points]
                    weather = weather_dict[state_dict['weather']]
                    new_state_dict = {'img_id': f'{img_path.stem}.jpg', 'human_num': len(points),
                                      'points': corrected_points, 'weather': weather}

                json_tar_dir = self.target_dir / stage / 'jsons' / f'{img_path.stem}.json'
                with json_tar_dir.open('w') as jsf:
                    new_state_dict = json.dumps(new_state_dict)
                    jsf.write(new_state_dict)

                print(f'{stage} set: [{i + 1} / {len(self.image_path_sets[stage])}] done. \r', end='')


class Integrator:
    def __init__(self):
        self.dataset_path_dict = {'nwpu': Path(r''),
                                  'ucf_qnrf': Path(r'')}

        self.target_path = Path(r'')
        self.stages = ['train', 'val', 'test']
        self.file_types = ['density_maps', 'images', 'jsons']
        for stage in self.stages:
            for file_type in self.file_types:
                (self.target_path / stage / file_type).mkdir(exist_ok=True)

    def process(self):
        def name_num_map(dataset, num):
            assert dataset in ['nwpu', 'ucf']
            if dataset == 'nwpu':
                if 3110 <= num <= 3609:
                    num += 367
                elif 2610 <= num <= 3109:
                    num += 1701
            elif dataset == 'ucf':
                if 1 <= num <= 867:
                    num += 2609
                elif 868 <= num <= 1201:
                    num += 3089
                elif 1202 <= num <= 1535:
                    num += 3609
            return num

        for dataset in ['nwpu', 'ucf_qnrf']:
            for stage in self.stages:
                for file_type in self.file_types:
                    file_paths = list((self.dataset_path_dict[dataset] / stage / file_type).glob('*'))
                    for file_path in file_paths:
                        num = int(file_path.name)
                        tar_num = name_num_map(dataset, num)
                        tar_name = f'{tar_num:04d}.{file_path.suffix}'
                        if file_type == 'jsons':
                            with file_path.open('r') as jsf:
                                state_dict = json.load(jsf)
                            state_dict['source'] = dataset
                            with (self.target_path / stage / file_type / tar_name).open('w') as jsf:
                                dict_s = json.dumps(state_dict)
                                jsf.write(dict_s)
                        else:
                            shutil.copy(file_path, self.target_path / stage / file_type / tar_name)
                        print(f'{dataset} {state_dict} {file_type} {file_path.name} done. ')


if __name__ == '__main__':
    collator = GCCCollator()
    # collator.process()
