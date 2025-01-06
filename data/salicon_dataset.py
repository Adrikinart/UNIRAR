from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torchvision import transforms
import numpy as np
import cv2
import PIL
import scipy.io
from .helper import normalize_tensor


class SALICONDataset(Dataset):

    def __init__(
            self,
            path,
            phase='train',
            out_size=(288, 384),
            target_size=(480, 640),
            preproc_cfg=None,
            limit = None,
            folder_img = 'images',
            folder_fix = 'fixations',
            folder_sal = 'maps',
            ):


        self.extensions = ['.png', '.jpg' , '.jpeg']
        self.folder_img = folder_img
        self.folder_fix = folder_fix
        self.folder_sal = folder_sal


        self.limit = limit
        self.phase = phase
        self.out_size = out_size
        self.target_size = target_size
        self.dir = Path(path) # path dataset

        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.phase_str = 'val' if phase in ('valid', 'eval') else phase
        self.file_stem = f"COCO_{self.phase_str}2014_"
        self.file_nr = "{:012d}"

        self.samples = self.prepare_samples()

    def get_map(self, img_nr):
        base_path = self.dir / self.folder_sal / (self.file_stem + self.file_nr.format(img_nr))
        
        for ext in self.extensions:
            map_file = Path(str(base_path) + ext)
            if map_file.exists():
                map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
                if map is not None:
                    return map
        
        raise FileNotFoundError(f"No valid saliency map found for image {img_nr} with extensions {self.extensions}")


    def get_img(self, img_nr):
        img_file = self.dir / self.folder_img / (self.file_stem + self.file_nr.format(img_nr))

        for ext in self.extensions:
            map_file = Path(str(img_file) + ext)
            if map_file.exists():
                img = cv2.imread(str(map_file))
                if img is not None:
                    return np.ascontiguousarray(img[:, :, ::-1])

        
        raise FileNotFoundError(f"No valid saliency map found for image {img_nr} with extensions {self.extensions}")

    def get_raw_fixations(self, img_nr):
        raw_fix_file = self.dir / self.folder_fix /  (self.file_stem + self.file_nr.format(img_nr) + '.mat')
        fix_data = scipy.io.loadmat(raw_fix_file)
        fixations_array = [gaze[2] for gaze in fix_data['gaze'][:, 0]]
        return fixations_array, fix_data['resolution'].tolist()[0]

    def process_raw_fixations(self, fixations_array, res):
        fix_map = np.zeros(res, dtype=np.uint8)
        for subject_fixations in fixations_array:
            fix_map[subject_fixations[:, 1] - 1, subject_fixations[:, 0] - 1]\
                = 255
        return fix_map

    def get_fixation_map(self, img_nr):
        fix_map_file = self.dir / self.folder_fix / (self.file_stem + self.file_nr.format(img_nr))

        for ext in self.extensions:
            map_file = Path(str(fix_map_file) + ext)
            if map_file.exists():
                fix_map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
                if fix_map is not None:
                    return fix_map

        fix_map_file = Path(str(fix_map_file) + ".png")
        fixations_array, res = self.get_raw_fixations(img_nr)
        fix_map = self.process_raw_fixations(fixations_array, res)
        cv2.imwrite(str(fix_map_file), fix_map)

        return fix_map

    def prepare_samples(self):
        samples = []

        for index , file in enumerate((self.dir / self.folder_img).glob(self.file_stem + '*.jpg')):
            samples.append(int(file.stem[-12:]))
        
        if self.limit is not None:
            samples= samples[:min(self.limit , len(samples))]
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        # if data == 'img':
        transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_nr):
        img = self.get_img(img_nr)
        img = self.preprocess(img, data='img')
        sal = self.get_map(img_nr)
        sal = self.preprocess(sal, data='sal')
        fix = self.get_fixation_map(img_nr)
        fix = self.preprocess(fix, data='fix')

        return img, sal, fix, self.out_size

    def __getitem__(self, item):
        img_nr = self.samples[item]
        return self.get_data(img_nr)

