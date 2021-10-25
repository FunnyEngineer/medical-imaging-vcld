from Preprocessor import *
from Trainer import *
from argparse import ArgumentParser
import torch, random
import numpy as np

SEED = 1119
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def main(args):
    p = Preprocessor(args)
    train_img, dev_img = p.preprocess()
    for path, mode in zip([f'{args.model_path}_1', f'{args.model_path}_2', f'{args.model_path}'], ['pretrain1', 'pretrain2', 'finetune']):
        print(f'In {mode} mode, the model will be saved at {path}')
        trainer = Trainer(train_img, dev_img, args, mode)
        trainer.training()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    main(args)
