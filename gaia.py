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
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import timeit
import re
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import stats
from file_io import *
from requirement import *
from policy import *
import wandb

# # of items in fashion coordination
NUM_ITEM_IN_COORDI = 4
NUM_ITEM_IN_COORDI_ZSL = 5
#  # of metadata features
NUM_META_FEAT = 4
# # of fashion coordination candidates
NUM_RANKING = 3
# image feature size
IMG_FEAT_SIZE = 2048
# augmentation ratio (aug2:aug1:org)
AUGMENTATION_RATIO = [0.5, 0.4, 0.1]

seed=2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

class FahsionHowDataset(Dataset):
    """ Fashion-How dataset."""
    def __init__(self, in_file_trn_dialog, swer, mem_size, emb_size,
                 crd_size, metadata, itm2idx, idx2itm, feats, num_rnk,
                 similarities, corr_thres, valid_num=0, is_valid=False):
        """
        initialize your data, download, etc.
        """
        self._swer = swer
        self._mem_size = mem_size
        self._emb_size = emb_size
        self._crd_size = crd_size
        self._metadata = metadata
        self._itm2idx = itm2idx
        self._idx2itm = idx2itm
        self._feats = feats
        self._num_rnk = num_rnk
        self._similarities = similarities
        self._corr_thres = corr_thres
        self._datatype = ['aug2', 'aug1', 'org']
        
        ''' 수정 사항 '''
        # make_io_trn_data는 deterministic 한 dataset 만드는 함수이므로 결과를 미리 저장하고 로드하는 것으로 바꿈
        self._dlg, self._crd = make_io_trn_data(
                        in_file_trn_dialog, self._swer, self._mem_size,
                        self._crd_size, self._itm2idx, self._idx2itm,
                        self._metadata, self._similarities, self._num_rnk,
                        self._corr_thres, self._feats)

        self._len = len(self._dlg)
        self._num_item_in_coordi = len(self._crd[0][0])

    def __getitem__(self, index):
        """
        get item
        """
        datatype = np.random.choice(self._datatype, 1, p=AUGMENTATION_RATIO)
        if datatype == 'aug2':
            crd = []
            crd.append(self._crd[index][0])
            for j in range(1, self._num_rnk):
                itm_lst = list(
                        permutations(np.arange(self._num_item_in_coordi), j))
                idx = np.arange(len(itm_lst))
                np.random.shuffle(idx)
                crd_new = replace_item(self._crd[index][0], self._itm2idx,
                                       self._idx2itm, self._similarities,
                                       itm_lst[idx[0]], self._corr_thres)
                crd.append(crd_new)
        elif datatype == 'aug1':
            crd = []
            for j in range(self._num_rnk - 1):
                crd.append(self._crd[index][j])
            idx = np.arange(self._num_item_in_coordi)
            np.random.shuffle(idx)
            crd_new = replace_item(crd[self._num_rnk-2], self._itm2idx,
                                   self._idx2itm, self._similarities,
                                   [idx[0]], self._corr_thres)
            crd.append(crd_new)
        else:
            crd =self._crd[index]
        # embedding
        vec_dialog = vectorize_dlg(self._swer, self._dlg[index])
        # memorize for end-to-end memory network
        mem_dialog = memorize_dlg(vec_dialog, self._mem_size, self._emb_size)
        # fashion item numbering
        idx_coordi = indexing_coordi_dlg(crd, self._crd_size, self._itm2idx)
        # convert fashion item to metadata
        vec_coordi = convert_dlg_coordi_to_metadata(idx_coordi,
                            self._crd_size, self._metadata, self._feats)
        return mem_dialog, vec_coordi

    def __len__(self):
        """
        return data length
        """
        return self._len


class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, emb_size, key_size, mem_size,
                 meta_size, hops, item_size,
                 coordi_size, eval_node, num_rnk,
                 use_batch_norm, use_dropout, zero_prob,
                 tf_dropout, tf_nhead,
                 tf_ff_dim, tf_num_layers,
                 use_multimodal, img_feat_size):
        """
        initialize and declare variables
        """
        super().__init__()
        # class instance for requirement estimation
        self._requirement = RequirementNet(emb_size, key_size,
                                    mem_size, meta_size, hops)
        # class instance for ranking
        self._policy = PolicyNet_(emb_size, key_size, item_size,
                meta_size, coordi_size, eval_node, num_rnk,
                use_batch_norm, use_dropout, zero_prob,
                tf_dropout, tf_nhead, tf_ff_dim, tf_num_layers,
                use_multimodal, img_feat_size)

    def forward(self, dlg, crd):
        """
        build graph
        """
        req = self._requirement(dlg)
        logits = self._policy(req, crd)
        return logits


class gAIa(object):
    """ Class for AI fashion coordinator """
    def __init__(self, args, device, name='gAIa'):
        """
        initialize
        """
        self.args = args
        self._device = device
        self._batch_size = args.batch_size
        self._model_path = args.model_path
        self._model_file = args.model_file
        self._epochs = args.epochs
        self._max_grad_norm = args.max_grad_norm
        self._save_freq = args.save_freq
        self._num_eval = args.evaluation_iteration
        use_dropout = args.use_dropout
        if args.mode == 'test' or args.mode == 'zsl':
            use_dropout = False

        # class instance for subword embedding
        swer = SubWordEmbReaderUtil(args.subWordEmb_path)
        self._emb_size = swer.get_emb_size()
        meta_size = NUM_META_FEAT
        # ZSL
        coordi_size = NUM_ITEM_IN_COORDI
        if args.mode == 'zsl':
            coordi_size = NUM_ITEM_IN_COORDI_ZSL
        feats_size = IMG_FEAT_SIZE
        self._num_rnk = NUM_RANKING
        self._rnk_lst = np.array(list(permutations(np.arange(self._num_rnk),
                                                   self._num_rnk)))

        # read metadata DB
        metadata, idx2item, item2idx, item_size, \
            similarities, feats = make_metadata(args.in_file_fashion,
                                        swer, coordi_size,
                                        meta_size, args.use_multimodal,
                                        args.in_dir_img_feats, feats_size)

        # prepare DB for training
        if args.mode == 'train':
            # dataloader
            train_dataset = FahsionHowDataset(args.in_file_trn_dialog, swer,
                                        args.mem_size, self._emb_size,
                                        coordi_size, metadata, item2idx,
                                        idx2item, feats, self._num_rnk,
                                        similarities, args.corr_thres, args.valid_num)
            
                
            self._num_examples = len(train_dataset)
            self._dataloader = DataLoader(train_dataset,
                                          batch_size=self._batch_size,
                                          shuffle=True, num_workers=8, pin_memory=True)
        # prepare DB for evaluation
        elif args.mode == 'test' or args.mode == 'zsl':
            self._tst_dlg, self._tst_crd = make_io_eval_data(
                    args.in_file_tst_dialog, swer, args.mem_size,
                    coordi_size, item2idx, metadata, self._num_rnk, feats)
            self._num_examples = len(self._tst_dlg)

        # model
        self._model = Model(self._emb_size, args.key_size, args.mem_size,
                            meta_size, args.hops, item_size, coordi_size,
                            args.eval_node, self._num_rnk, args.use_batch_norm,
                            use_dropout, args.eval_zero_prob, args.tf_dropout,
                            args.tf_nhead, args.tf_ff_dim, args.tf_num_layers,
                            args.use_multimodal, feats_size)

        print('\n<model parameters>')
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)

        if args.mode == 'train':
            # optimizer
            self._optimizer = optim.SGD(self._model.parameters(), lr=args.learning_rate)
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self._optimizer, T_0=20, T_mult=2, eta_min=0)

    def _get_loss(self, batch, loss_fct):
        """
        calculate loss
        """
        dlg, crd = batch
        crd_shuffle = []
        rnk_shuffle = []
        for c in crd:
            rnk_rnd, crd_rnd = shuffle_one_coordi_and_ranking(
                                    self._rnk_lst, c, self._num_rnk)
            crd_shuffle.append(torch.stack(crd_rnd))
            rnk_shuffle.append(torch.tensor(rnk_rnd))
        crd = torch.stack(crd_shuffle)
        rnk = torch.stack(rnk_shuffle)
        dlg = dlg.type(torch.float32)
        crd = crd.type(torch.float32)
        logits = self._model(dlg, crd)
        loss = 0.0

        # label =  torch.Tensor(rnk).long().to(self._device) # (n,6)
        # for i in range(len(logits)):
        #     loss += loss_fct(logits[i], label[i])
        print(logits.shape)
        logits = torch.mean(logits, dim=1)
        print(logits.shape)
        for i in range(len(logits)):
            probs = nn.functional.softmax(logits[i], dim=0)
            for j in range(len(self._rnk_lst)):
                corr, _ = stats.weightedtau(
                                self._num_rnk-1-self._rnk_lst[rnk[i]],
                                self._num_rnk-1-self._rnk_lst[j])
                loss += ((1.0 - corr) * 0.5 * probs[j])

        label =  torch.Tensor(rnk).long().to(self._device) # (n,6)
        preds = torch.argmax(logits, 1)
        return loss, preds, label

    def train(self):
        """
        training
        """
        #wandb.init(project="fasionhow-task4",config=self.args) # , entity="stitching-tailors"
        wandb.init(project="task4", entity="stitching-tailors",config=self.args)
        wandb.run.name = 'clip_loss'
        loss_fct = torch.nn.CrossEntropyLoss()

        print('\n<Train>')
        print('total examples in dataset: {}'.format(self._num_examples))
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)
        init_epoch = 1
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))
                init_epoch += int(re.findall('\d+', file_name)[-1])
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        self._model.to(self._device)
        end_epoch = self._epochs + init_epoch
        for curr_epoch in range(init_epoch, end_epoch):
            calc_train_acc = torchmetrics.Accuracy()
            time_start = timeit.default_timer()
            losses = []
            
            iter_bar = tqdm(self._dataloader)
            for batch_idx, batch in enumerate(iter_bar):
                example_ct = curr_epoch * (len(self._dataloader)) + batch_idx
                self._optimizer.zero_grad()
                batch = [t.to(self._device) for t in batch]
                loss, preds, labels = self._get_loss(batch, loss_fct)#.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(),
                                         self._max_grad_norm)
                self._optimizer.step()
                losses.append(loss.mean())
                train_acc = calc_train_acc(preds.cpu().detach(), labels.cpu().detach())
            time_end = timeit.default_timer()
            
            print('-'*30)
            print('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            print('Time: {:.2f}sec'.format(time_end - time_start))
            print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('Val pred:', preds.detach().cpu(), 'label:', labels.detach().cpu())
            print("train_acc: ", calc_train_acc.compute())
            print('-'*30)
            #self._lr_scheduler.step()
            wandb.log({"train_loss":float(torch.mean(torch.tensor(losses)).detach().cpu()),  \
                       "train_acc":calc_train_acc.compute(), \
                       'learning_rate':float(self._lr_scheduler.get_last_lr()[0])}, step=example_ct+1)
            if curr_epoch % self._save_freq == 0:
                file_name = os.path.join(self._model_path,
                                         'gAIa-{}.pt'.format(curr_epoch))
                torch.save({'model': self._model.state_dict()}, file_name)
        print('Done training; epoch limit {} reached.\n'.format(self._epochs))
        return True

    def _calculate_weighted_kendal_tau(self, pred, label):
        """
        calcuate Weighted Kendal Tau Correlation
        """
        total_count = 0
        total_corr = 0
        for p, l in zip(pred, label):
            corr, _ = stats.weightedtau(
                            self._num_rnk-1-self._rnk_lst[l],
                            self._num_rnk-1-self._rnk_lst[p])
            total_corr += corr
            total_count += 1
        return (total_corr / total_count)

    def _predict(self, eval_dlg, eval_crd):
        """
        predict
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        eval_crd = torch.tensor(eval_crd).to(self._device)
        preds = []
        for start in range(0, eval_num_examples, self._batch_size):
            end = start + self._batch_size
            if end > eval_num_examples:
                end = eval_num_examples
            _, pred = self._model(eval_dlg[start:end],
                                  eval_crd[start:end])
            pred = pred.cpu().numpy()
            for j in range(end-start):
                preds.append(pred[j])
        preds = np.array(preds)
        return preds, eval_num_examples

    def _evaluate(self, eval_dlg, eval_crd):
        """
        evaluate
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_corr = []
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        for i in range(self._num_eval):
            preds = []
            # DB shuffling
            coordi, rnk = shuffle_coordi_and_ranking(eval_crd, self._num_rnk)
            coordi = torch.tensor(coordi).to(self._device)
            for start in range(0, eval_num_examples, self._batch_size):
                end = start + self._batch_size
                if end > eval_num_examples:
                    end = eval_num_examples
                _, pred = self._model(eval_dlg[start:end],
                                      coordi[start:end])
                pred = pred.cpu().numpy()
                for i in range(end-start):
                    preds.append(pred[i])
            preds = np.array(preds)
            # compute Weighted Kendal Tau Correlation
            corr = self._calculate_weighted_kendal_tau(preds, rnk)
            eval_corr.append(corr)
        return np.array(eval_corr), eval_num_examples

    # 결과 생성
    def zsl(self):
        """
        create prediction.csv
        """
        print('\n<Predict>')

        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # predict
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        # 실제 제출결과 생성시 경로는 '/home/work/model/prediction.csv'로 고정
        np.savetxt("/home/work/model/prediction.csv", preds.astype(int), encoding='utf8', fmt='%d')
        time_end = timeit.default_timer()
        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('-'*50)

    def test(self):
        """
        test
        """
        print('\n<Test>')

        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # evluation
        test_corr, num_examples = self._evaluate(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*30)
        print('Test Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('Test WKTC: {:.4f}'.format(np.mean(test_corr)))
        print('-'*30)
