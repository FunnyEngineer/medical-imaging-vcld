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

Q = [[0.85, 0.87, 0.86, 0.76, 0.95],
    [0.85, 0.86, 0.86, 0.74, 0.94],
    [0.83, 0.86, 0.84, 0.74, 0.95],
    [0.84, 0.89, 0.83, 0.74, 0.94],
    [0.85, 0.87, 0.86, 0.74, 0.94], # resnet_ccl
    [0.85, 0.89, 0.83, 0.77, 0.96], # resnet_div
    [0.86, 0.86, 0.84, 0.76, 0.92], # resnet_div2
    [0.84, 0.88, 0.86, 0.76, 0.94], # resnet_div3
    [0.85, 0.86, 0.82, 0.76, 0.95], # resnet_div4
    [0.84, 0.87, 0.8, 0.77, 0.96], # resnet_div5
    [0.85, 0.86, 0.85, 0.76, 0.95], # resnet_div7
    [0.85, 0.87, 0.84, 0.74, 0.95], # resnet_div8
    [0.84, 0.87, 0.8, 0.77, 0.96], # resnet_all
    [0.85, 0.86, 0.87, 0.74, 0.95], # resnet_byol
    [0.85, 0.87, 0.86, 0.75, 0.94], # resnet_mask
    [0.86, 0.87, 0.81, 0.74, 0.95], # resnet_label
    [0.84, 0.86, 0.85, 0.75, 0.95], # resnet_cnn
    [0.85, 0.85, 0.85, 0.73, 0.95], # resnet_final2
    [0.85, 0.87, 0.86, 0.74, 0.95], # resnet_final3
    [0.85, 0.86, 0.82, 0.74, 0.94], # resnet_final4
    [0.85, 0.86, 0.85, 0.74, 0.95], # resnet_final5
    ]

outfile = ['rnn_aux_aug.csv', 'rnn_aux_aug2.csv', 'rnn_aux_aug3.csv', 'cv.csv', 'ccl.csv', 'div.csv', 'div2.csv', 'div3.csv', 'div4.csv', 'div5.csv', 
            'div7.csv', 'div8.csv', 'all.csv', 'byo.csv', 'mask.csv', 'label.csv', 'cnn.csv', 'final2.csv', 'final3.csv', 'best.csv', 'final5.csv']

models = ['resnet_div', 'resnet_div2', 'resnet_div3', 'resnet_div4', 'resnet_div5', 'resnet_div7', 'resnet_div8', 'resnet_all', 'resnet_byol',
        'resnet_mask', 'resnet_label', 'resnet_cnn', 'resnet_final2', 'resnet_final3', 'resnet_final4', 'resnet_final5']    

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
    # model.load_state_dict(torch.load(args.model_path, map_location=torch.device(f"cuda:{CUDA_NUMBER}")))
    testset = Dataset_patient_based(test_img, test=1)
    testloader = DataLoader(testset, shuffle=False, batch_size=4, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=testset.collate_fn)

    for j in range(5, len(Q)):
        model = None
        # input('3')
        if j <= 15:
            model = Extractor2(j).cuda(CUDA_NUMBER)
        elif j == 16:
            model = Extractor3().cuda(CUDA_NUMBER)
        else:
            model = Extractor().cuda(CUDA_NUMBER)
        model.load_state_dict(torch.load(os.path.join(args.model_path, models[j - 5]), map_location=torch.device(f"cuda:{CUDA_NUMBER}")))
        print(models[j - 5], Q[j])
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
                if j <= 16:
                    aux_pred, pred = model(img, None, None, mask, None, None, test=1)
                else:
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
        quan = Q[j]
        threshold = np.array([np.quantile(Pred[:, i], quan[i]) for i in range(NUM_CLASS)])
        yhat = (Pred > threshold).astype(np.int32)
        print(outfile[j], fbeta_score(Label, yhat, beta=2, average='micro'))
        df = {'dirname': dirname, 'ID': ID}
        for i in range(yhat.shape[1]):
            key = classes[i]
            df[key] = yhat[:, i]
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(args.csv_path, outfile[j]), index=False)
        to_kaggle(os.path.join(args.csv_path, outfile[j]), os.path.join(args.csv_path, outfile[j]))

def ensemble(path, files, output_file):
    tmp = []
    for file in files:
        filepath = os.path.join(path, file)
        df = pd.read_csv(filepath)
        tmp.append(df['prediction'])
        # print(file, sum(tmp[-1]))
    pred = tmp[0]
    for i in range(1, len(tmp)):
        pred += tmp[i]
    majority = len(files) // 2
    pred = [1 if i > majority else 0 for i in pred]
    ID = df['ID']
    df = pd.DataFrame({'ID':ID, 'prediction':pred})
    df.to_csv(os.path.join(path, output_file), index=False)
    return df 

def Ensemble(path):
    ensemble(path, ['rnn_aux_aug.csv', 'rnn_aux_aug2.csv', 'cv.csv'], 'ensemble4.csv')
    ensemble(path, ['rnn_aux_aug.csv', 'rnn_aux_aug2.csv', 'rnn_aux_aug3.csv', 'cv.csv', 'div.csv'], 'ensemble5.csv')
    ensemble(path, ['rnn_aux_aug2.csv', 'rnn_aux_aug3.csv', 'div.csv'], 'ensemble6.csv')
    ensemble(path, ['rnn_aux_aug2.csv', 'rnn_aux_aug3.csv', 'div2.csv', 'div3.csv', 'div4.csv'], 'ensemble8.csv')
    ensemble(path, ['div2.csv', 'div5.csv', 'all.csv'], 'ensemble9.csv')
    ensemble(path, ['div2.csv', 'div3.csv', 'div5.csv', 'byo.csv', 'all.csv'], 'ensemble10.csv')
    ensemble(path, ['div2.csv', 'div5.csv', 'label.csv'], 'ensemble14.csv')
    ensemble(path, ['cnn.csv', 'div5.csv', 'label.csv'], 'ensemble15.csv')
    ensemble(path, ['cnn.csv', 'ccl.csv', 'label.csv'], 'ensemble18.csv')
    ensemble(path, ['cnn.csv', 'ccl.csv', 'label.csv', 'div2.csv', 'div5.csv'], 'ensemble19.csv')
    ensemble(path, ['div8.csv', 'ccl.csv', 'label.csv'], 'ensemble21.csv')
    ensemble(path, ['final2.csv', 'final3.csv', 'label.csv'], 'ensemble24.csv')
    ensemble(path, ['best.csv', 'final5.csv', 'label.csv'], 'ensemble26.csv')

    ensemble(path, ['div2.csv', 'ensemble5.csv', 'ensemble6.csv'], 'ensemble7.csv')
    ensemble(path, ['ensemble7.csv', 'ensemble9.csv', 'ensemble10.csv'], 'ensemble11.csv')
    ensemble(path, ['ensemble4.csv', 'ensemble5.csv', 'ensemble6.csv', 'ensemble8.csv', 'ensemble9.csv', 'ensemble10.csv',
             'ensemble14.csv', 'ensemble15.csv', 'div2.csv', 'div5.csv', 'div7.csv', 'label.csv', 'cnn.csv', 'all.csv', 'mask.csv'], 'ensemble17.csv')
    ensemble(path, ['ensemble11.csv', 'ensemble18.csv', 'ensemble19.csv'], 'ensemble20.csv')
    ensemble(path, ['ensemble11.csv', 'ensemble20.csv', 'ensemble21.csv'], 'ensemble23.csv')
    ensemble(path, ['ensemble20.csv', 'ensemble23.csv', 'ensemble24.csv'], 'ensemble25.csv')
    ensemble(path, ['ensemble17.csv', 'ensemble25.csv', 'ensemble26.csv'], 'ensemble.csv')


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
    if args.predict == 1:
        p = Preprocessor(args)
        test_img = p.preprocess(test=1)
        predict(args, test_img)
    Ensemble(args.csv_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("predict", type=int)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    main(args)
