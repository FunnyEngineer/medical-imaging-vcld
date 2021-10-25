from Dataset import *
from torch.utils.data import DataLoader
from Config import *
from Model import *
import torch.nn as nn
import pickle, time
from torchvision import transforms
import numpy as np
from itertools import chain
from sklearn.metrics import fbeta_score
from losses import *
from Preprocessor import *


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def save(arr, filename):
  with open(filename, 'wb') as f:
    pickle.dump(arr, f)

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

def f2_loss(pred, target):
  pred = torch.sigmoid(pred)
  tp = (pred * target).sum(0)
  tn = ((1-pred) * (1-target)).sum(0)
  fp = (pred * (1-target)).sum(0)
  fn = ((1-pred) * target).sum(0)
  p = tp / (tp + fp + 1e-20)
  r = tp / (tp + fn + 1e-20)

  f2 = (5 * p * r) / (4 * p + r + 1e-20)
  return 1 - f2.mean()

weights = torch.FloatTensor([7.54204026, 11.21974831, 7.64145131, 5.07824726, 24.03321765]).cuda(CUDA_NUMBER)

class Trainer(object): # micro F1
    def __init__(self, train_img, dev_img, args, mode):
        ## train_img is dictionary, key is patient, 
        ## train_img[key] is a list, element is a (image, label, filename) tuple
        self.train = train_img
        self.dev = dev_img
        self.mode = mode
        print(f'CUDA_NUMBER = {CUDA_NUMBER}')
        self.model = Extractor().cuda(CUDA_NUMBER)
        if self.mode == 'pretrain2':
          self.model.load_state_dict(torch.load(f"{args.model_path}_1", map_location=torch.device(f"cuda:{CUDA_NUMBER}")))
        if self.mode == 'finetune':
          self.model.load_state_dict(torch.load(f"{args.model_path}_2", map_location=torch.device(f"cuda:{CUDA_NUMBER}")))
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.wbce = nn.BCEWithLogitsLoss(weight=weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.asy = AsymmetricLossOptimized()
        self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.37, 0.7446, 0.9144, 0.976, 0.9952, 0.9997]).cuda(CUDA_NUMBER))
        self.args = args
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=LR_DECAY, patience=LR_PATIENCE, mode='min')
        
    def load_part_model(self, model_path):
      c = 0
      own_state = self.model.state_dict()
      state_dict = torch.load(model_path, map_location=torch.device(f"cuda:{CUDA_NUMBER}"))
      for name, param in state_dict.items():
        if isinstance(param, nn.Parameter):
          parameter = param
          param = param.data
        if name in own_state and param.shape == own_state[name].shape:
          own_state[name].copy_(param)
          own_state[name].requires_grad = False
          c += 1
      #     print(name, c, own_state[name].requires_grad)

    def training(self):    
      trainset = Dataset_patient_based(self.train)
      trainloader = DataLoader(trainset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=trainset.collate_fn)
      devset = Dataset_patient_based(self.dev, test=1)
      devloader = DataLoader(devset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, worker_init_fn=worker_init_fn, collate_fn=devset.collate_fn)
      img_pairs = torch.load('./sample.pt')
      pairset = Pairs_Dataset(img_pairs)
      pairloader = DataLoader(pairset, shuffle=True, batch_size=8)
      Best_F2, epochs, c, Best_loss, Best_loss2 = 0, 0, 0, 100000, 100000
      Byol_loss = 0
      if self.mode != 'pretrain1':
        epsilon = 0.5
      else:
        epsilon = 5
      while True:
        epochs += 1
        steps, Total, Aux_Loss, Loss = 0, 0, 0, 0
        Acc, NCE, Count_Loss, CCL_Loss = 0, 0, 0, 0
        self.model.train()
        for i, (img, label, filename, mask) in enumerate(trainloader):
          img = img.cuda(CUDA_NUMBER)
          mask = mask.cuda(CUDA_NUMBER)
          img = img.repeat(1, 1, 3, 1, 1)

          bsize, seq_len = img.shape[0], img.shape[1]
          n = int(seq_len * P)
          n2 = int(seq_len * P)
          masked_mask = None
          if np.random.uniform(size=1)[0] >= epsilon and n2 >= 1:
            lens = mask.sum(1).cpu()
            masked_idx = torch.LongTensor(np.random.choice(seq_len, size=(bsize, n2), replace=False))
            masked_mask = torch.zeros(bsize, seq_len).scatter(1, masked_idx, torch.ones(bsize, seq_len)).bool() # [2, 40]
            masked_mask = masked_mask.cuda(CUDA_NUMBER)
          if np.random.uniform(size=1)[0] >= epsilon and n >= 1:
            img1, img2, pair_label = pairloader.__iter__().next()
            img1 = img1[:n].repeat(1, 3, 1, 1).cuda(CUDA_NUMBER) # [6, 3, 512, 512]
            img2 = img2[:n].repeat(1, 3, 1, 1).cuda(CUDA_NUMBER) # [6, 3, 512, 512]
            pair_label = pair_label[:n].unsqueeze(0).expand(bsize, -1, NUM_CLASS) # [2, 6, 5]
            change_idx = torch.LongTensor(np.random.choice(seq_len, size=(bsize, n), replace=False))
            change_mask = torch.zeros(bsize, seq_len).scatter(1, change_idx, torch.ones(bsize, seq_len)).bool() # [2, 40]
            change_label = label.masked_select(change_mask.unsqueeze(-1)).reshape(bsize, n, NUM_CLASS)
            change_label += pair_label
            repeat_mask = (change_label > 1).sum(dim=2).bool()
            label.masked_scatter_(change_mask.unsqueeze(-1), change_label)

            change_mask, repeat_mask = change_mask.cuda(CUDA_NUMBER), repeat_mask.cuda(CUDA_NUMBER)
            aux_pred, pred, acc, nce, label_pred, ccl_pred, lcc = self.model(img, img1, img2, mask, change_mask, repeat_mask, masked_mask)
          else:
            aux_pred, pred, acc, nce, label_pred, ccl_pred, lcc = self.model(img, None, None, mask, None, None, masked_mask)
          
          label = label.clamp(max=1).reshape(-1, NUM_CLASS).cuda(CUDA_NUMBER)
          mask = mask.reshape([-1, 1]).bool().bitwise_not()
          
          ''' NCE '''
          NCE += nce.item()
          Acc += acc.item()

          ''' CCL '''
          ccl_pred = ccl_pred.masked_fill(mask, -1e12)
          # ccl_loss = self.wbce(ccl_pred, label) + lcc * CCL_LAMBDA + f2_loss(ccl_pred, label)
          ccl_loss = self.asy(ccl_pred, label) + lcc * CCL_LAMBDA
          # ccl_loss = self.wbce(ccl_pred, label) + lcc * CCL_LAMBDA
          # ccl_loss = f2_loss(ccl_pred, label) * 100 + lcc * CCL_LAMBDA
          CCL_Loss += ccl_loss.item()

          ''' Label Prediction '''
          label_pred = label_pred.masked_fill(mask, -1e12)
          label_count = label.sum(1).long()
          count_loss = self.ce(label_pred, label_count)
          Count_Loss += count_loss.item()

          aux_pred = aux_pred.masked_fill(mask, -1e12)
          #aux_loss = self.bce(aux_pred, label)
          # aux_loss = f2_loss(aux_pred, label) * 10
          aux_loss = self.asy(aux_pred, label) + f2_loss(aux_pred, label) * 10
          # aux_loss = self.wbce(aux_pred, label) + f2_loss(aux_pred, label)
          Aux_Loss += aux_loss.item()

          pred = pred.masked_fill(mask, -1e12)
          loss = self.asy(pred, label) + f2_loss(pred, label) * 10
          # loss = self.wbce(pred, label) + f2_loss(pred, label)
          # print(self.wbce(pred, label).item(), f2_loss(pred, label).item())
          #loss = f2_loss(pred, label)
          # loss = self.asy(pred, label)
          Loss += loss.item()
          
          if self.mode != 'finetune':
            Total += (loss.item() + aux_loss.item())
            loss = loss + aux_loss
          else:
            Total += (loss.item() + aux_loss.item() + nce.item() + count_loss.item() + ccl_loss.item())
            loss = loss + aux_loss + nce + count_loss + ccl_loss

          # Total += loss.item()
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

          steps += 1
          if steps == 1 or steps % LOG_EVERY_STEP == 0:
            print(f'Epochs = {epochs}, Steps = {steps}, Total = {round(Total / steps, 4)}, Loss = {round(Loss / steps, 4)}, Aux_Loss = {round(Aux_Loss / steps, 4)}, NCE = {round(NCE / steps, 4)}, Acc = {round(Acc / steps, 4)}, Count_Loss = {round(Count_Loss / steps, 4)}, CCL_Loss = {round(CCL_Loss / steps, 4)}, LR = {self.optim.param_groups[0]["lr"]}')

        print(f'Epochs = {epochs}, Steps = {steps}, Total = {round(Total / steps, 4)}, Loss = {round(Loss / steps, 8)}, Aux_Loss = {round(Aux_Loss / steps, 8)}, NCE = {round(NCE / steps, 4)}, Acc = {round(Acc / steps, 4)}, Count_Loss = {round(Count_Loss / steps, 4)}, CCL_Loss = {round(CCL_Loss / steps, 4)}, LR = {self.optim.param_groups[0]["lr"]}')

        steps, Total, Aux_Loss, Loss, Count_Loss = 0, 0, 0, 0, 0
        Total2 = 0
        Acc, NCE, Count_Loss, CCL_Loss = 0, 0, 0, 0
        Pred, Label, Mask = [], [], []
        self.model.eval()
        with torch.no_grad():
          for i, (img, label, filename, mask) in enumerate(devloader):
            img = img.cuda(CUDA_NUMBER)
            label = label.reshape(-1, NUM_CLASS).cuda(CUDA_NUMBER)
            mask = mask.cuda(CUDA_NUMBER)
            img = img.repeat(1, 1, 3, 1, 1)

            aux_pred, pred, acc, nce, label_pred, ccl_pred, lcc = self.model(img, None, None, mask, None, None, None)
            mask = mask.reshape([-1, 1]).bool().bitwise_not()

            ''' NCE '''
            NCE += nce.item()
            Acc += acc.item()

            ''' CCL '''
            ccl_pred = ccl_pred.masked_fill(mask, -1e12)
            # ccl_loss = self.wbce(ccl_pred, label) + lcc * CCL_LAMBDA + f2_loss(ccl_pred, label)
            ccl_loss = self.asy(ccl_pred, label) + lcc * CCL_LAMBDA
            # ccl_loss = self.wbce(ccl_pred, label) + lcc * CCL_LAMBDA
            # ccl_loss = f2_loss(ccl_pred, label) * 100 + lcc * CCL_LAMBDA
            CCL_Loss += ccl_loss.item()

            ''' Label Prediction '''
            label_pred = label_pred.masked_fill(mask, -1e12)
            label_count = label.sum(1).long()
            count_loss = self.ce(label_pred, label_count)
            Count_Loss += count_loss.item()

            aux_pred = aux_pred.masked_fill(mask, -1e12)
            #aux_loss = self.bce(aux_pred, label)
            # aux_loss = f2_loss(aux_pred, label) * 10
            aux_loss = self.asy(aux_pred, label) + f2_loss(aux_pred, label) * 10
            # aux_loss = self.wbce(aux_pred, label) + f2_loss(aux_pred, label)
            Aux_Loss += aux_loss.item()

            pred = pred.masked_fill(mask, -1e12)
            loss = self.asy(pred, label) + f2_loss(pred, label) * 10
            # loss = self.wbce(pred, label) + f2_loss(pred, label)
            # loss = self.asy(pred, label)
            Loss += loss.item()
            
            if self.mode != 'finetune':
              Total += (loss.item() + aux_loss.item())
            else:
              Total += (loss.item() + aux_loss.item() + count_loss.item() + ccl_loss.item() + nce.item())
            # Total += loss.item()

            Pred.append(pred)
            Label.append(label)
            Mask.append(mask)
            steps += 1

            if steps == 1 or steps % LOG_EVERY_STEP == 0:
              print(f'Epochs = {epochs}, Steps = {steps}, Total = {Total / steps}, Validation Loss = {Loss / steps}, Aux_Loss = {Aux_Loss / steps}, NCE = {round(NCE / steps, 4)}, Acc = {round(Acc / steps, 4)},  Count_Loss = {round(Count_Loss / steps, 4)}, CCL_Loss = {round(CCL_Loss / steps, 4)}')

        Pred = torch.cat(Pred, dim=0).cpu().numpy()
        Label = torch.cat(Label, dim=0).cpu().numpy()
        Mask = torch.cat(Mask, dim=0).cpu().numpy()
        Mask = np.logical_not(Mask).squeeze(1)
        f2_score, threshold = f2(Pred[Mask], Label[Mask])
        print(f'Epochs = {epochs}, Steps = {steps}, Total = {Total / steps}, Validation Loss = {Loss / steps}, Aux_Loss = {Aux_Loss / steps}, NCE = {round(NCE / steps, 4)}, Acc = {round(Acc / steps, 4)},  Count_Loss = {round(Count_Loss / steps, 4)}, CCL_Loss = {round(CCL_Loss / steps, 4)}, F2 = {round(f2_score, 8)}, Threshold = {threshold}')
        Total /= steps
        # Total2 /= steps

        # if Total2 < Best_loss2:
        #   Best_F2_2 = f2_score
        #   Best_loss2 = Total2
        #   best_epochs2 = epochs
        #   torch.save(self.model.state_dict(), f'{self.args.model_path}_2')
        #   print(f'Best F2 = {Best_F2_2}, Loss = {Total2}, Threshold = {threshold}')

        if Total < Best_loss:
          Best_F2 = f2_score
          Best_loss = Total
          best_epochs = epochs
          if self.mode == 'pretrain1':
            torch.save(self.model.state_dict(), f'{self.args.model_path}_1')
          elif self.mode == 'pretrain2':
            torch.save(self.model.state_dict(), f'{self.args.model_path}_2')
          elif self.mode == 'finetune':
            torch.save(self.model.state_dict(), f'{self.args.model_path}')
          c = 0
          print(f'Best F2 = {Best_F2}, Loss = {Total}, Threshold = {threshold}')

        else:
          print(f'Does not Improve!!! Best Loss = {Best_loss}, Best F2 = {Best_F2}, Loss = {Total}, Best_epoch = {best_epochs}')
          c += 1
        self.lr_scheduler.step(Total)
        if self.optim.param_groups[0]["lr"] < 1e-6:
          break
    
