from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from config import BertConfig
from bert_models import BERT, PreTrainedBertModel, BertLMPredictionHead, TransformerBlock, gelu, Attention, LayerNorm
import dill

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class TSNE(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(TSNE, self).__init__(config)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc

        freeze_afterwards(self)

    def forward(self, output_dir, output_file='graph_embedding.tsv'):
        # dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.embedding
        # rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.embedding

        if not self.config.graph:
            print('save embedding not graph')
            rx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(3, len(self.rx_voc.word2idx) + 3, dtype=torch.long))
            dx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(len(self.rx_voc.word2idx) + 3, len(self.rx_voc.word2idx) + 3 + len(self.dx_voc.word2idx),
                             dtype=torch.long))
        else:
            print('save embedding graph')

            dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.get_all_graph_emb()
            rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.get_all_graph_emb()

        np.savetxt(os.path.join(output_dir, 'dx-' + output_file),
                   dx_graph_emb.detach().numpy(), delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'rx-' + output_file),
                   rx_graph_emb.detach().numpy(), delimiter='\t')

        # def dump(prefix='dx-', emb):
        #     with open(prefix + output_file ,'w') as fout:
        #         m = emb.detach().cpu().numpy()
        #         for
        #         fout.write()


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHead(config, dx_voc_size), ClsHead(
            config, dx_voc_size), ClsHead(config, rx_voc_size), ClsHead(config, rx_voc_size)])

    def forward(self, dx_inputs, rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), self.multi_cls[1](rx_inputs), self.multi_cls[2](dx_inputs), self.multi_cls[3](rx_inputs)


class GBERT_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(GBERT_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx) # 1997
        self.rx_voc_size = len(rx_voc.word2idx) # 468

        self.bert = BERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, rx_labels=None):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(rx2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)


class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)


class GBERT_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer):
        super(GBERT_Predict, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(5*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        self.bertconv1d = nn.Conv1d(in_channels=110,
                                    out_channels=config.hidden_size,
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    dilation=1) # padding = (kernel_size - 1) * dilation
        self.interval_emb = nn.Linear(in_features=1, out_features=config.hidden_size)
        self.pre_embedding = nn.Linear(55, config.hidden_size)
        self.post_embedding = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.qkv_attention = Attention()

        self.cnn_norm = LayerNorm(config.hidden_size)
        self.qkv_norm = LayerNorm(config.hidden_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # self.apply(self.init_bert_weights)


    def forward(self, input_ids, interval_ids, dx_labels=None, rx_labels=None, epoch=None):
        """
        :param input_ids: [B, max_seq_len]
        :param interval_ids: [B, max_seq_len]
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        # input_emb_block
        input_fc = self.pre_embedding(input_ids.float())
        input_pool = input_fc.view(2, -1, input_fc.size(1))
        dx_input_pool = input_pool[0]  # (adm, H)
        rx_input_pool = input_pool[1]
        dxrx_input_emb = []
        for i in range(rx_labels.size(0)):
            dx_input_mean = torch.mean(dx_input_pool[0:i + 1, :], dim=0, keepdim=True)
            rx_input_mean = torch.mean(rx_input_pool[0:i + 1, :], dim=0, keepdim=True)
            dxrx_input_emb.append(torch.cat([dx_input_mean, rx_input_mean], dim=-1).cpu().detach().numpy())
        dxrx_input_emb = np.array(dxrx_input_emb)
        dxrx_input_emb = torch.from_numpy(dxrx_input_emb).cuda()
        dxrx_input_emb = self.post_embedding(dxrx_input_emb)

        # interval_gate
        lambda_t = self.sigmoid(dxrx_input_emb)

        # interval_block
        interval_x = self.interval_emb(interval_ids)
        interval_x = self.relu(interval_x)
        g_t = self.tanh(torch.div(1, torch.log(math.e + interval_x)))
        sin_t = torch.sin(interval_x)
        discount_t = lambda_t * g_t + (1 - lambda_t) * sin_t
        discount_t = dxrx_input_emb * discount_t
        qkvatt = self.qkv_attention(dxrx_input_emb, discount_t, discount_t)[0]
        qkvatt = self.qkv_norm(qkvatt)

        #bert_block
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # cnn_block
        input_ids_new = input_ids.view(-1, 110)
        input_ids_new = self.bertconv1d(torch.unsqueeze(input_ids_new.float(), dim=0).permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)
        input_ids_new = self.cnn_norm(input_ids_new)

        # cnn_bert_interval_catblock
        rx_logits = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            ids_mean = torch.mean(input_ids_new[0:i+1, :], dim=0, keepdim=True)
            # print(rx_mean.size(), ids_mean.size())
            # concat
            #, qkvatt[i]
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0), ids_mean, qkvatt[i]], dim=-1)
            rx_logits.append(self.cls(concat))
        rx_logits = torch.cat(rx_logits, dim=0)



        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)

        loss_ddi = ddi_loss(torch.sigmoid(rx_logits), 0.5, path='../data/ddi_A_final.pkl')
        loss += loss_ddi * 0.1
        return loss, rx_logits

def ddi_loss(pred_y, threshould, path='../data/ddi_A_final.pkl'):
        ddi_A = dill.load(open(path, 'rb'))
        # pred_y, torch tensor, shape=(1, med_num)
        #print(type(ddi_A))
        # print(ddi_A.shape)
        y_pred_tmp = pred_y.clone().detach().cpu().numpy()[0]
        y_pred_tmp[y_pred_tmp >= threshould] = 1
        y_pred_tmp[y_pred_tmp < threshould] = 0
        idx_pred = list(np.where(y_pred_tmp == 1)[0])

        device = torch.device('cuda')
        ddi_loss = torch.tensor(0.0).to(device)
        cnt = 0
        for i, med_i in enumerate(idx_pred):
            for j, med_j in enumerate(idx_pred):
                if j < i:
                    continue
                cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    ddi_loss += pred_y[0, med_i] * pred_y[0, med_j]
        if len(idx_pred) > 0:
            ddi_loss /= cnt
        return ddi_loss

class GBERT_Predict_Side(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, side_len):
        super(GBERT_Predict_Side, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        self.side = nn.Sequential(nn.Linear(
            side_len, side_len // 2), nn.ReLU(), nn.Linear(side_len // 2, side_len // 2))
        self.final_cls = nn.Sequential(nn.ReLU(), nn.Linear(len(
            tokenizer.rx_voc_multi.word2idx) + side_len // 2, len(tokenizer.rx_voc_multi.word2idx)))
        # self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, rx_labels=None, epoch=None, input_sides=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :param input_side: [adm-1, side_len]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        visit_vecs = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
            concat_trans = self.cls(concat)
            visit_vecs.append(concat_trans)

        visit_vecs = torch.cat(visit_vecs, dim=0)
        # add side and concat
        side_trans = self.side(input_sides)
        patient_vec = torch.cat([visit_vecs, side_trans], dim=1)

        rx_logits = self.final_cls(patient_vec)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits

# ------------------------------------------------------------
