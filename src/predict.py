from Config import *
from Model import *
from Dataset import *
from Preprocessor import *
import torch, random, csv, subprocess
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import fbeta_score, multilabel_confusion_matrix

SEED = 1119
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def f2(pred, target):
  T = np.zeros(5)
  q = np.zeros(5)
  for i in range(5):
      candidate = []
      for j in range(101):
          threshold = np.quantile(pred[:, i], j / 100)
          T[i] = threshold
          q[i] = j / 100
          yhat = (pred > T).astype(np.int32)
          score = fbeta_score(target, yhat, beta=2, average='micro')
          candidate.append((score, q.copy(), T.copy()))
      assert len(candidate) == 101
      score, q, T = max(candidate, key=lambda x: x[0])
  return score, q

def predict(args, test_img):
    classes = ['ich', 'ivh', 'sah', 'sdh', 'edh']
    model = Extractor().cuda(CUDA_NUMBER)
    model.load_state_dict(torch.load("./src/best_model", map_location=torch.device(f"cuda:{CUDA_NUMBER}")))
    testset = Dataset_patient_based(test_img, test=1)
    testloader = DataLoader(testset, shuffle=False, batch_size=4, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=testset.collate_fn)

    Pred, Label, Mask = [], [], []
    Aux_Pred = []
    dirname = []
    ID = []
    model.eval()
    steps = 0
    with torch.no_grad():
        for i, (img, label, filename, mask) in tqdm(enumerate(testloader)):
            img = img.cuda(CUDA_NUMBER)
            label = label.reshape(-1, NUM_CLASS).cuda(CUDA_NUMBER)
            mask = mask.cuda(CUDA_NUMBER)
            img = img.repeat(1, 1, 3, 1, 1)
            aux_pred, pred, acc, nce, label_pred, ccl_pred, lcc, cnn, rnn = model(img, None, None, mask, None, None, None, test=1)
            
            Aux_Pred.append(aux_pred)
            Pred.append(pred)
            Label.append(label)
            Mask.append(mask.reshape(-1,).bool())
            dirname.extend([j.split('/')[0] for j in filename if j != ''])
            ID.extend([j.split('/')[1] for j in filename if j != ''])

        Aux_Pred = torch.cat(Aux_Pred, dim=0).cpu().numpy()
        Pred = torch.cat(Pred, dim=0).cpu().numpy()
        Label = torch.cat(Label, dim=0).cpu().numpy()
        Mask = torch.cat(Mask, dim=0).cpu().numpy()
        Aux_Pred = Aux_Pred[Mask]
        Pred = Pred[Mask]
        Label = Label[Mask]
        quan = [0.85, 0.86, 0.82, 0.74, 0.94]
        threshold = np.array([np.quantile(Pred[:, i], quan[i]) for i in range(NUM_CLASS)])
        yhat = (Pred > threshold).astype(np.int32)
        print(fbeta_score(Label, yhat, beta=2, average='micro'))
        df = {'dirname': dirname, 'ID': ID}
        for i in range(yhat.shape[1]):
            key = classes[i]
            df[key] = yhat[:, i]
        df = pd.DataFrame(df)
        df.to_csv(args.output_path, index=False)
        to_kaggle(args.output_path, args.output_path)

def to_kaggle(inpath, outpath):
    kaggle_eval_classes = ['ich', 'ivh', 'sah', 'sdh', 'edh']
    with open(inpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        output_rows = []
        for row_idx, row in enumerate(csv_reader):
            if row_idx == 0:
                continue
            for cls_idx, cls in enumerate(kaggle_eval_classes):
                ID_single = row[1].split('.')[0] + '_' + cls
                output_row = [ID_single, row[cls_idx + 2]]
                output_rows.append(output_row)

    with open(outpath, mode='w') as csv_file:
        fieldnames = ['ID', 'prediction']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in output_rows:
            writer.writerow({'ID': row[0], 'prediction': row[1]})

def main(args):
    p = Preprocessor(args)
    test_img = p.preprocess(test=1)
    predict(args, test_img)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(args)
