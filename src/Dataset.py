import torch, os
from torchvision import transforms
import torch.utils.data as data
from Config import *
from torch.utils.data.sampler import Sampler
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from util import *

filenameToPILImage = lambda x: Image.open(x)

class Dataset_img_based(data.Dataset):
    def __init__(self, img_dict, test=0):
        self.test = test
        self.img_dict = img_dict
        self.X = [i[0] for key in tqdm(img_dict.keys()) for i in img_dict[key]]
        self.Y = torch.FloatTensor([i[1] for key in tqdm(img_dict.keys()) for i in img_dict[key]])
        self.filename = [i[2] for key in tqdm(img_dict.keys()) for i in img_dict[key]]
        # self.transform = transforms.Compose([
        #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.159], [0.2979]) #512
        #     ])
        # self.rotation = transforms.Compose([
        #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #     transforms.RandomRotation(10),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.159], [0.2979])
        # ])
        # self.flip = transforms.Compose([
        #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #     transforms.RandomHorizontalFlip(p=1),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.159], [0.2979])
        # ])

        self.resize = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ])
        self.aug = [transforms.Compose([transforms.RandomRotation(10)]),
                    transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
                    transforms.Compose([transforms.RandomResizedCrop(512)])
                    # MultiScaleCrop(IMG_SIZE, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2)
                    ]
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.159], [0.2979])
        ])
        print(f'Got {len(self.X)} images')
    
    def __getitem__(self, idx):
        if self.test == 0:
            aug = np.random.randint(0, len(self.aug) + 1, size=1)[0]
            img = self.resize(self.X[idx])
            if aug > 0:
                aug_seq = np.random.choice(len(self.aug), size=aug, replace=False)
                for j in aug_seq:
                    img = self.aug[j](img)
            img = self.totensor(img)
            return img, self.Y[idx], self.filename[idx]
        else:
            return self.transform(self.X[idx]), self.Y[idx], self.filename[idx]

    def __len__(self):
        return len(self.X)

class Pairs_Dataset(data.Dataset):
    def __init__(self, img_pairs):
        self.img1, self.img2 = img_pairs
        self.Y = [torch.FloatTensor([1, 0, 0, 0, 0]).repeat(828).reshape(-1 ,NUM_CLASS),
                  torch.FloatTensor([0, 1, 0, 0, 0]).repeat(555).reshape(-1 ,NUM_CLASS),
                  torch.FloatTensor([0, 0, 1, 0, 0]).repeat(723).reshape(-1 ,NUM_CLASS),
                  torch.FloatTensor([0, 0, 0, 1, 0]).repeat(661).reshape(-1 ,NUM_CLASS),
                  torch.FloatTensor([0, 0, 0, 0, 1]).repeat(200).reshape(-1 ,NUM_CLASS)
                ]
        self.Y = torch.cat(self.Y, dim=0)
        print(self.img1.shape, self.img2.shape, self.Y.shape)
        assert self.Y.shape[0] == self.img1.shape[0] == self.img2.shape[0]

    def __getitem__(self, idx):
        return self.img1[idx], self.img2[idx], self.Y[idx]

    def __len__(self):
        return len(self.img1)

class Dataset_patient_based(data.Dataset):
    def __init__(self, img_dict, test=0):
        self.img_dict = img_dict
        self.patient_id = list(self.img_dict.keys())
        self.X = [[] for _ in range(len(self.patient_id))]
        self.Y = [[] for _ in range(len(self.patient_id))]
        self.filename = [[] for _ in range(len(self.patient_id))]
        for idx, key in tqdm(enumerate(img_dict.keys())):
            assert key == self.patient_id[idx]
            for img in img_dict[key]:
                self.X[idx].append(img[0])
                self.Y[idx].append(img[1])
                self.filename[idx].append(img[2])
            self.Y[idx] = torch.FloatTensor(self.Y[idx])
        self.max_img_seq = max([len(self.img_dict[key]) for key in self.img_dict.keys()])

        self.resize = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ])
        # self.aug = [transforms.Compose([transforms.RandomRotation()]),
        #             transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
        #             #transforms.Compose([transforms.RandomResizedCrop(512)])
        #             MultiScaleCrop(IMG_SIZE, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2)
        #             ]
        self.aug = [transforms.Compose([transforms.RandomRotation(45)]),
                    transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
                    transforms.Compose([transforms.RandomAffine(45)]),
                    transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
                    transforms.Compose([transforms.ColorJitter(brightness=1)]),
                    transforms.Compose([transforms.ColorJitter(contrast=1)]),
                    #transforms.Compose([transforms.RandomResizedCrop(512)])
                    MultiScaleCrop(IMG_SIZE, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                    # Cutout(n_holes=1, length=16),
                    # RandomErasing()
                    ]
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.159], [0.2979])
        ])

        self.test = test
        self.warp = Warp(512)
        print(f'Got {len(self.X)} sequences')
    
    def __getitem__(self, idx):
        pid = self.patient_id[idx]
        #aug = np.random.randint(0, 3, size=len(self.X[idx]))
        aug = np.random.randint(0, len(self.aug) + 1, size=len(self.X[idx]))
        X = []
        j = 0
        for i, (img, aug_idx) in enumerate(zip(self.X[idx], aug)):
            img = self.resize(img)
            # if self.test == 0 and aug_idx < 3:
            #     img = self.aug[aug_idx](img)
            if self.test == 0 and aug_idx > 0:
                aug_seq = np.random.choice(len(self.aug), size=aug_idx, replace=False)
                for j in aug_seq:
                    if j >= 7:
                        break
                    else:
                        img = self.aug[j](img)
            elif self.test == 1:
                self.warp(img)
            img = self.totensor(img)
            if j >= 7:
                img = self.aug[j](img)
            X.append(img)

        X = torch.stack(X, dim=0)
        Y = self.Y[idx]
        filename = self.filename[idx]
        mask = [1 for _ in range(X.shape[0])]
        mask = torch.LongTensor(mask)
        return X, Y, filename, mask

    def __len__(self):
        return len(self.patient_id)

    def collate_fn(self, data):
        data = sorted(data, key=lambda x: x[0].shape[0], reverse=True)
        X = [i[0] for i in data]
        Y = [i[1] for i in data]
        mask = [i[3] for i in data]
        filename = []
        max_seq = max([i.shape[0] for i in mask])
        for i in range(len(X)):
            npad = max_seq - X[i].shape[0]
            filename.extend(data[i][2])
            if npad != 0:
                pad_img = torch.zeros((npad, 1, IMG_SIZE, IMG_SIZE)).float()
                pad_label = torch.zeros((npad, 5)).long()
                X[i] = torch.cat([X[i], pad_img], dim=0)
                Y[i] = torch.cat([Y[i], pad_label], dim=0)
                mask[i] = torch.cat([mask[i], torch.zeros(npad)], dim=0)
                filename += [""] * npad
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)
        mask = torch.stack(mask, dim=0)
        #print(X.shape, Y.shape, mask.shape, len(filename))

        return X, Y, filename, mask





