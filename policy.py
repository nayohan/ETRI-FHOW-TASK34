'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class PolicyNet_(nn.Module):
    """Class for policy network"""
    def __init__(self, emb_size, key_size, item_size, meta_size, 
                 coordi_size, eval_node, num_rnk, use_batch_norm, 
                 use_dropout, eval_zero_prob, tf_dropout, tf_nhead, 
                 tf_ff_dim, tf_num_layers, use_multimodal,
                 img_feat_size, name='PolicyNet'):
        """
        initialize and declare variables
        """
        super().__init__()
        self._item_size = item_size
        self._emb_size = emb_size
        self._key_size = key_size
        self._meta_size = meta_size
        self._coordi_size = coordi_size
        self._num_rnk = num_rnk
        self._name = name

        buf = eval_node[1:-1]
        buf = list(map(int, buf.split(',')))
        self._eval_out_node = buf[0]
        self._num_hid_rnk = buf[1:]
        self._num_hid_layer_rnk = len(self._num_hid_rnk)
        
        self._count_eval = 0
        if use_dropout:
            dropout = tf_dropout
        else:
            dropout = 0.0
            eval_zero_prob = 0.0
        num_heads = tf_nhead
        num_layers = tf_num_layers
        dim_ff = tf_ff_dim

        num_in = self._emb_size * self._meta_size 
        if use_multimodal:
            num_in += img_feat_size            
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=num_in, dim_feedforward=dim_ff, 
                dropout=dropout, nhead=num_heads)
        self._transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)    
        self._summary = nn.Linear(num_in, self._eval_out_node)            
        self._queries = nn.Linear(self._key_size, num_in)            
        
        mlp_rnk_list = OrderedDict([])
        num_in = self._eval_out_node * self._num_rnk + self._key_size
        for i in range(self._num_hid_layer_rnk+1):
            if i == self._num_hid_layer_rnk:
                num_out = math.factorial(self._num_rnk)
                mlp_rnk_list.update({ 
                    'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
            else:
                num_out = self._num_hid_rnk[i]
                mlp_rnk_list.update({ 
                    'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
                mlp_rnk_list.update({
                    'layer%s_relu'%(i+1): nn.ReLU()})
                if use_batch_norm:
                    mlp_rnk_list.update({
                    'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})
                if use_dropout:
                    mlp_rnk_list.update({
                    'layer%s_dropout'%(i+1): nn.Dropout(p=eval_zero_prob)})
            self._count_eval += (num_in * num_out + num_out)
            num_in = num_out
        self._mlp_rnk = nn.Sequential(mlp_rnk_list) 

        self._clip_crd = nn.Linear(self._eval_out_node, 100)
        # self._clip_crd2 = nn.Linear(self._eval_out_node, 100)        
        # self._clip_crd3 = nn.Linear(self._eval_out_node, 100)
        self._clip_dlg1 = nn.Linear(self._eval_out_node, 100)
        self._clip_dlg2 = nn.Linear(self._eval_out_node, 100)
        self._clip_dlg3 = nn.Linear(self._eval_out_node, 100)
        self._output_fc  = nn.Linear(9, 6)
        
    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """
        bat_size = crd.size()[0]
        num_in = crd.size()[2]
        queries = self._queries(req)
        queries = torch.reshape(queries, (bat_size, 1, num_in))
        inputs = torch.cat((queries, crd), dim=1)
        inputs = torch.transpose(inputs, 0, 1)
        enc = self._transformer(inputs)
        enc_m = torch.mean(enc, dim=0)
        evl = self._summary(enc_m)
        return evl
    
    def _ranking_coordi(self, in_rnk):
        """
        rank candidates         
        """
        out_rnk = self._mlp_rnk(in_rnk)
        return out_rnk
    
    def _coordi_encode(self, crd):
        #print('crd:', crd.shape) # ([b, 4, 2560])
        inputs = torch.transpose(crd, 0, 1) # ([4, b, 2560])
        enc = self._transformer(inputs)
        enc_m = torch.mean(enc, dim=0) # ([b, 2560])
        #print('enc_m:', enc_m.shape)
        evl = self._summary(enc_m) # ([b, 600])
        evl = torch.unsqueeze(evl, 1) # ([b, 1, 600])
        #print('evl:', evl.shape)
        return evl
        
    def forward(self, req, crd):
        """
        build graph for evaluation and ranking         
        """
        dlg_emb1 = self._clip_dlg1(req) # (b, 3, 300) -> (b, 3, 100)
        dlg_emb2 = self._clip_dlg2(req) # (b, 3, 100)
        dlg_emb3 = self._clip_dlg3(req) # (b, 3, 100)
        dlg_emb = torch.stack([dlg_emb1, dlg_emb2, dlg_emb3], dim=1) #([b, 3, 100])
        crd_tr = torch.transpose(crd, 1, 0) # (4,3,2560)
        for i in range(self._num_rnk):
            crd_eval = self._coordi_encode(crd_tr[i])
            if i == 0:
                crd_emb = crd_eval
            else:
                crd_emb = torch.cat((crd_emb, crd_eval), 1)
                
        crd_end = self._clip_crd(crd_emb) # ( b, 3, 100)
        crd_end = torch.transpose(crd_end, 1, 2) # (b, 100, 3)

        similarity = (100.0 * torch.bmm(dlg_emb, crd_end))#.softmax(dim=0) # (b, 3, 100), (b,100, 3) -> (b, 3, 3)
        #similarity = torch.reshape(similarity, (-1, 9))
        #out_rnk = self._output_fc(similarity)
        #soft_out_rnk = out_rnk.softmax(dim=1)
        #print('similarity.shape:',  similarity.shape) # ([3, 3])
        #print('similarity:',  similarity) # ([3, 3])
        #print('out_rnk:',  out_rnk) # ([3, 3])
        #$print('soft_out_rnk:', soft_out_rnk)
        return similarity
        
