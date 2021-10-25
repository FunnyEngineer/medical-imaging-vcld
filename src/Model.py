import torch
import torch.nn.functional as F
import torch.nn as nn
from Config import *
import torchvision.models as models
from math import floor

# VGG16: 1531
# Densenet121: 1043
# Alexnet: 1245
# Googlenet: 1029
# ResNet-18: 1063
# EfficientNet: 1033
# MobileNet: 1025

class Extractor2(nn.Module):
    def __init__(self, idx):
        super(Extractor2, self).__init__()
        self.idx = idx
        modules = list(models.resnet18(pretrained=True).children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.hidden_size = 512
        
        self.seq = nn.GRU(self.hidden_size, self.hidden_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, NUM_CLASS)
        )
        if idx == 15:
            self.label_prediction = nn.Sequential(
                nn.Linear(self.hidden_size, NUM_CLASS + 1)
            )
    
        self.init_weight()

    def init_weight(self):
        for m in self._modules['classifier']:
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x, pair1, pair2, mask, change_mask, repeat_mask, test=0):
        # duplicate_mask [4, 40]
        #print(x.shape, pair1.shape, pair2.shape)
        bsize, seq = x.shape[0], x.shape[1]
        x = x.view(-1, 3, IMG_SIZE, IMG_SIZE)
        features = self.extractor(x)
        features = features.view(bsize, seq, -1)
        if pair1 != None:
            n = pair1.shape[0]
            pair1_features = self.extractor(pair1).view(n, -1).unsqueeze(0).expand(bsize, n, -1) # [2, 4, 512]
            pair2_features = self.extractor(pair2).view(n, -1).unsqueeze(0).expand(bsize, n, -1) # [2, 4, 512]

            candidate = features.masked_select(change_mask.unsqueeze(-1)).reshape(pair1_features.shape[0], pair1_features.shape[1], -1)
            diff = (pair2_features - pair1_features).masked_fill(repeat_mask.unsqueeze(-1), 0)
            out = candidate + diff
            features.masked_scatter(change_mask.unsqueeze(-1), out)
        
        features = self.dropout(features)
        
        lens = mask.sum(1).cpu()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True)
        hout, hn = self.seq(rnn_inputs)
        hout, lens = nn.utils.rnn.pad_packed_sequence(hout, batch_first=True)

        pred = self.classifier(hout)
        pred = pred.reshape(-1, NUM_CLASS)

        aux_pred = self.classifier(features)
        aux_pred = aux_pred.reshape(-1, NUM_CLASS)
        return aux_pred, pred

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        modules = list(models.resnet18(pretrained=True).children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.hidden_size = 512
        
        self.seq = nn.GRU(self.hidden_size, self.hidden_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        # self.classifier = nn.Sequential(
        #     # nn.Linear(hidden_size, hidden_size),
        #     # nn.LeakyReLU(0.2, inplace=True), 
        #     # nn.Dropout(0.5),
        #     # nn.Linear(hidden_size, hidden_size),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Dropout(0.5),
        #     nn.Linear(self.hidden_size, NUM_CLASS)
        # )
        self.label_prediction = nn.Sequential(
            nn.Linear(self.hidden_size, NUM_CLASS + 1)
        )
        
        self.ccl = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, NUM_CLASS) # [5, 512])
        nn.init.kaiming_normal_(self.classifier.weight)

        self.Wk = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(TIMESTEP)])
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.Wk.apply(self.init_weight2)

        self.init_weight()

    def init_weight(self):
        for m in self._modules['label_prediction']:
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def init_weight2(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def l2_norm(self, x):
        # [batch_size, hidden_dim]
        size = x.size()
        buffer = torch.pow(x, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        output = torch.div(x, norm.view(-1, 1))
        output = output.view(size)

        return output

    def forward(self, x, pair1, pair2, mask, change_mask, repeat_mask, masked_mask, test=0):
        # duplicate_mask [4, 40]
        #print(x.shape, pair1.shape, pair2.shape)
        bsize, seq = x.shape[0], x.shape[1]
        x = x.view(-1, 3, IMG_SIZE, IMG_SIZE)
        features = self.extractor(x)
        features = features.view(bsize, seq, -1)

        if masked_mask != None:
            masked_features = features.masked_fill(masked_mask.unsqueeze(-1), 0)
            features = self.dropout(masked_features)
        else:
            features = self.dropout(features)
        
        if pair1 != None:
            n = pair1.shape[0]
            pair1_features = self.extractor(pair1).view(n, -1).unsqueeze(0).expand(bsize, n, -1) # [2, 4, 512]
            pair2_features = self.extractor(pair2).view(n, -1).unsqueeze(0).expand(bsize, n, -1) # [2, 4, 512]

            candidate = features.masked_select(change_mask.unsqueeze(-1)).reshape(pair1_features.shape[0], pair1_features.shape[1], -1)
            diff = (pair2_features - pair1_features).masked_fill(repeat_mask.unsqueeze(-1), 0)
            out = candidate + diff
            features.masked_scatter(change_mask.unsqueeze(-1), out)
            features = self.dropout(features)
        
        lens = mask.sum(1).cpu()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True)
        hout, hn = self.seq(rnn_inputs)
        hout, lens = nn.utils.rnn.pad_packed_sequence(hout, batch_first=True)

        pred = self.classifier(hout)
        pred = pred.reshape(-1, NUM_CLASS)

        aux_pred = self.classifier(features)
        aux_pred = aux_pred.reshape(-1, NUM_CLASS)
        rnn = hout
        cnn = features
        ''' Label Prediction Start '''
        label_pred = self.label_prediction(hout)
        label_pred = label_pred.reshape(-1, NUM_CLASS + 1)
        ''' Label Prediction End '''

        ''' CCL Start '''
        e = self.ccl(features).reshape(-1, self.hidden_size)
        c = self.classifier.weight.unsqueeze(0).expand(e.shape[0], NUM_CLASS, -1).reshape(-1, self.hidden_size)
        e = e.unsqueeze(1).expand(e.shape[0], NUM_CLASS, -1).reshape(-1, self.hidden_size)
        
        diff = self.l2_norm(e) - self.l2_norm(c)
        dist = torch.pow(diff, 2).sum(1).reshape(bsize, seq, -1)
        ccl_pred = -dist.reshape(-1, NUM_CLASS)

        k1 = self.classifier.weight.unsqueeze(1).expand(NUM_CLASS, NUM_CLASS, -1).reshape(-1, self.hidden_size)
        k2 = self.classifier.weight.unsqueeze(0).expand(NUM_CLASS, NUM_CLASS, -1).reshape(-1, self.hidden_size)
        diff = self.l2_norm(k1) - self.l2_norm(k2)
        diff = torch.pow(diff, 2).sum(1)
        lcc = F.relu(diff - B).mean()
        ''' CCL End '''

        ''' NCE Start '''
        nce, correct = 0, 0
        encode_samples = torch.empty((TIMESTEP, bsize, self.hidden_size)).float()
        t_samples = torch.randint(seq - TIMESTEP, size=(1,)).long() # randomly pick time stamps
        mask = mask[:, :t_samples + 1]
        for i in range(1, TIMESTEP + 1):
            encode_samples[i-1] = features[:, t_samples+i, :].view(-1, self.hidden_size) # [4, 512]
        lens = mask.sum(1).cpu()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True)
        hout, hn = self.seq(rnn_inputs)
        hout, lens = nn.utils.rnn.pad_packed_sequence(hout, batch_first=True)
        c_t = hout[:, t_samples, :].view(bsize, self.hidden_size) # [4, 512]
        mask = mask[:, t_samples].bool().reshape(-1)
        invmask = mask.logical_not().cuda(CUDA_NUMBER) # padding is True
        pred2 = torch.empty((TIMESTEP, bsize, self.hidden_size)).float()
        for i in range(TIMESTEP):
            pred2[i] = self.Wk[i](c_t)

        for i in range(TIMESTEP):
            total = torch.mm(encode_samples[i], pred2[i].transpose(0,1)) # [4, 4]
            correct += torch.eq(F.softmax(total, dim=0).argmax(dim=0), torch.arange(0, bsize)).masked_select(mask).sum()
            loss = torch.diag(self.lsoftmax(total)).cuda(CUDA_NUMBER).masked_fill(invmask, 0).sum()
            # total = self.lsoftmax(total).cuda(CUDA_NUMBER).masked_fill(invmask.unsqueeze(-1), 0)
            # mean = (total.sum(0) - torch.diag(total)) / (mask.sum() - 1 + 1e-20)
            # loss = (torch.diag(total) - mean).masked_fill(invmask, 0).sum()
            nce += loss
        bsize = mask.sum()
        nce /= (-bsize * TIMESTEP)
        accuracy = correct.item() / (bsize * TIMESTEP)
        ''' NCE END '''

        if test == 1:
            return aux_pred, pred, accuracy, nce, label_pred, ccl_pred, lcc, cnn, rnn
        else:
            return aux_pred, pred, accuracy, nce, label_pred, ccl_pred, lcc

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class Extractor3(nn.Module):

    def __init__(self):
        super(Extractor3, self).__init__()

        modules = list(models.resnet18(pretrained=True).children())[:-1]
        self.extractor = nn.Sequential(*modules)

        self.hidden_size = 512
        self.conv = nn.ModuleList()
        # filter_sizes = "3,5,7,9,11,13,15,17".split(',')
        filter_sizes = "3,5,7,11,13,15".split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            nn.init.xavier_uniform_(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = [self.hidden_size, NUM_FILTER_MAPS]
            for idx in range(1):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True, 0.2)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.dropout = nn.Dropout(0.5)
        self.seq = nn.GRU(self.hidden_size, self.hidden_size // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, NUM_CLASS)
        )

        # self.U = nn.Linear(self.filter_num * NUM_FILTER_MAPS + self.hidden_size, self.hidden_size)
        # nn.init.xavier_uniform_(self.U.weight)
        
        self.U = nn.Linear(self.filter_num * NUM_FILTER_MAPS + self.hidden_size, NUM_CLASS)
        # self.leakyrelu = nn.LeakyReLU(0.2)
        # self.U2 = nn.Linear(self.hidden_size, NUM_CLASS)
        # nn.init.xavier_uniform_(self.U2.weight)
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules['classifier']:
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
        # nn.init.xavier_uniform_(self.U.weight)

    def forward(self, x, pair1, pair2, mask, change_mask, repeat_mask, test=0):
        bsize, seq = x.shape[0], x.shape[1]
        x = x.view(-1, 3, IMG_SIZE, IMG_SIZE)
        features = self.extractor(x)
        features = features.view(bsize, seq, -1)
        
        if pair1 != None:
            n = pair1.shape[0]
            pair1_features = self.extractor(pair1).view(n, -1).unsqueeze(0).expand(bsize, n, -1) # [2, 4, 512]
            pair2_features = self.extractor(pair2).view(n, -1).unsqueeze(0).expand(bsize, n, -1) # [2, 4, 512]

            candidate = features.masked_select(change_mask.unsqueeze(-1)).reshape(pair1_features.shape[0], pair1_features.shape[1], -1)
            diff = (pair2_features - pair1_features).masked_fill(repeat_mask.unsqueeze(-1), 0)
            out = candidate + diff
            features.masked_scatter(change_mask.unsqueeze(-1), out)
        features = self.dropout(features)

        ''' RNN PART '''
        lens = mask.sum(1).cpu()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True)
        hout, hn = self.seq(rnn_inputs)
        hout, lens = nn.utils.rnn.pad_packed_sequence(hout, batch_first=True)

        ''' CNN PART '''
        x = features.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        
        x = torch.cat([hout, x], dim=2)
        # x = self.U(x)
        # x += hout
        # pred = self.classifier(x)
        pred = self.U(x)
        # pred = self.U2(self.leakyrelu(pred))
        pred = pred.view(-1, NUM_CLASS)

        aux_pred = self.classifier(features)
        aux_pred = aux_pred.reshape(-1, NUM_CLASS)
        return aux_pred, pred