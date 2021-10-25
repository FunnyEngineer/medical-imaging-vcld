import os, torch, pickle, time
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load(pathname):
    start = time.time()
    with open(pathname, "rb") as f:
        img = pickle.load(f)
    print(f'Load from {pathname}, Spend {round(time.time() - start, 4)}')
    return img

filenameToPILImage = lambda x: Image.open(x)
class Preprocessor(object):
    def __init__(self, args):
        self.args = args
        # self.transform = transforms.Compose([
        #     filenameToPILImage,
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
    
    def preprocess(self, test=0):
        if test == 0:
            train_patient = os.listdir(os.path.join(self.args.data_path, "train"))
            train_patient, dev_patient = train_test_split(train_patient, test_size=0.1, random_state=1119)
            train_csv = pd.read_csv(os.path.join(self.args.data_path, "train.csv"))
            img2label = self.map_img_to_label(train_csv)
            train_img = self.get_img(train_patient, os.path.join(self.args.data_path, "train"), img2label)
            dev_img = self.get_img(dev_patient, os.path.join(self.args.data_path, "train"), img2label)
            return train_img, dev_img
        if test == 1:
            test_patient = os.listdir(os.path.join(self.args.data_path, "test"))
            test_img = self.get_img(test_patient, os.path.join(self.args.data_path, "test"), None)
            return test_img

        # print('Start Loading Images')
        # start = time.time()
        # if test == 0:
        #     train_img = load(os.path.join(self.args.data_path, "train.pkl"))
        #     dev_img = load(os.path.join(self.args.data_path, "dev.pkl"))
        #     return train_img, dev_img
        # else:
        #     test_img = load(os.path.join(self.args.data_path, "test.pkl"))
        #     return test_img
        # print(f"Loading Finished, Spend {round(time.time() - start, 4)}")
        
    def map_img_to_label(self, df):
        img2label = {}
        for i in df.iterrows():
            label = list(i[1][-5:])
            img2label[i[1][1]] = label
        return img2label

    def get_img(self, x, path, img2label):
        img_dict = {}
        c = 0
        for pid in tqdm(x):
            filename = os.listdir(os.path.join(path, pid))
            filename = sorted(filename, key=lambda x: int(x.split('.')[0].split('_')[1]))
            img_dict[pid] = []
            for i in filename:
                pathname = os.path.join(path, pid, i)
                with Image.open(pathname) as img:
                    if img2label != None:
                        img_dict[pid].append((img.copy(), img2label[i], f'{pid}/{i}'))
                    else:
                        img_dict[pid].append((img.copy(), [0, 0, 0, 0, 0], f'{pid}/{i}'))
                c += 1
        print(f"Total {len(img_dict)} Patient and {c} images in {path}")
        return img_dict
