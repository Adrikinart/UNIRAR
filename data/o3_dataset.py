from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import PIL
import os 
import csv 
import random
from .helper import normalize_tensor


class O3Dataset(Dataset):
    n_train_val_images = 64
    dynamic = False

    def __init__(
            self,
            path,
            preproc_cfg=None,
            input_size = (412,412),
            target_size = (412,412), 
        ):

        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)

        self.dir = Path(path)
        self.img_size = input_size
        self.target_size = target_size

        self.load_data()
        random.shuffle(self.data)

        
    def load_data(self):
        self.data = []

        with open(os.path.join(self.dir , 'image_properties.csv'), 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            # self.data_csv['fields'] = next(csvreader)
            for row in csvreader:
                if os.path.isfile(self.dist_dir / row[0]) == False:
                    continue
                if os.path.isfile(self.img_dir / row[0]) == False:
                    continue
                if os.path.isfile(self.target_dir / row[0]) == False:
                    continue

                self.data.append(row)
            print("Total no. of rows: %d" % (csvreader.line_num))

    def get_dist(self, img_idx):
        map_file = self.dist_dir / self.data[img_idx][0]
        return cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)

    def get_img(self, img_idx):
        img_file = self.img_dir / self.data[img_idx][0]
        return np.ascontiguousarray(cv2.imread(str(img_file))[:, :, ::-1])

    def get_target(self, img_idx):
        fix_map_file = self.target_dir / self.data[img_idx][0]
        return cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)

    @property
    def dist_dir(self): return self.dir / 'dist_labels'
    @property
    def target_dir(self): return self.dir / 'targ_labels'
    @property
    def img_dir(self): return self.dir / 'images'

    def __len__(self):
        return len(self.data)

    def preprocess(self, img, out_size=None, data='img'):
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(out_size, interpolation=PIL.Image.LANCZOS if data != 'fix' else PIL.Image.NEAREST), transforms.ToTensor()
            ]
        if data == 'img':
            transformations.append(transforms.Normalize(
                self.preproc_cfg['rgb_mean'],
                self.preproc_cfg['rgb_std']
                ))
        # else:
            # transformations.append(transforms.Lambda(normalize_tensor))
        return transforms.Compose(transformations)(img)

    def get_data(self, img_idx):

        img = self.preprocess(self.get_img(img_idx), out_size=self.img_size)
        targ = self.preprocess(self.get_target(img_idx), out_size=self.target_size, data='sal')
        dist = self.preprocess(self.get_dist(img_idx), out_size=self.target_size, data='fix')

        return img, targ, dist, self.target_size

    def __getitem__(self, idx):
        return self.get_data(idx)
