import argparse
import os
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import random
from code.dataloader import TrainDataset
from code.evaluation import batch_evaluation
from code.model import BGE_Encoder, L2
from code.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training and Testing BGE Model on Large-Scale Graphs')
    parser.add_argument('--dataset', default='MAG', type=str)
    parser.add_argument('--data_root_dir', default='./dataset', type=str, help='root path of datasets')
    parser.add_argument('--model_root_dir', default='./models', type=str, help='root path for saving models')
    parser.add_argument('--test_batch', default=100, type=int)
    parser.add_argument('--topk', default='10,20', type=str, help='top-K recommendation setting')
    parser.add_argument('--n_negs', default=10, type=int, help='number of negative samples')
    parser.add_argument('--loss', default='mini_batch_CE', type=str, help='full_CE, mini_batch_CE')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--batch_per_epoch', default=10000, type=int, help='max training batch per epoch')
    parser.add_argument('--full_batch', default=1, type=int, help='whether to train all edges in an epoch')
    parser.add_argument('--reg', default=0.002, type=float, help='regularization coefficient')
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--gpu', default='0', type=str, help='gpu number')  # set this rather than CUDA_VISIBLE_DEVICES
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--test_batch_interval', default=200, type=int, help='train xx batches before evaluation')  # the main difference in options between large and small datasets
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_wo_test', default=False, action='store_true', help='save model without test')  # save model only
    parser.add_argument('--overwrite_model', default=False, action='store_true', help='only keep the best model')
    parser.add_argument('--do_test', default=1, type=int)
    parser.add_argument('--print_max_K', default=1, type=int, help='only print metrics@K for the max(topk)')
    parser.add_argument('--dim', default=64, type=int, help='embedding dimensionality')
    parser.add_argument('--n_anchor', default=16, type=int, help='number of anchor nodes')

    args = parser.parse_args()
 
    args.input_dir = os.path.join(args.data_root_dir, args.dataset)
 
    args.topk = list(map(int, args.topk.split(',')))
    args.max_K = max(args.topk)
    args.target_metric = f'ndcg@{args.max_K}'
 
    if args.loss == 'full_CE':
        args.n_negs = 0
  
    if args.save_model or args.save_model_wo_test:
        args.model_dir = os.path.join(args.model_root_dir, args.dataset)
        os.makedirs(args.model_dir, exist_ok=True)
 
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    if args.gpu == '-1':
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0'
    
    set_random_seed(args.seed, device != 'cpu')
    
    # load graphs: train, test
    print('load dataset...')
    s = time.time()
    with open(os.path.join(args.input_dir, 'train.csr.pickle'), 'rb') as f:
        csr_train = pickle.load(f)
        print('train:', csr_train.shape, csr_train.nnz)
    with open(os.path.join(args.input_dir, 'test.csr.pickle'), 'rb') as f:
        csr_test = pickle.load(f)
        print('test:', csr_test.shape, csr_test.nnz)
    e = time.time()
    print("loading time: %f s" % (e - s))
    
    # model
    num_U, num_V = csr_train.shape
    model = BGE_Encoder(dim=args.dim, num_U=num_U, num_V=num_V, n_anchor=args.n_anchor, dim_anchor=args.n_anchor//2)
    if device != 'cpu':
        model = model.cuda()
    
    max_epoch = args.max_epoch
    
    batch_per_epoch = args.batch_per_epoch
    full_batch = args.full_batch
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    regularizer = L2(args.reg)
    print('construct model done.')

    # dataloader
    print('construct dataloader...')
    train_dataloader = DataLoader(
            TrainDataset(csr_train, args.n_negs), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )
    
    # train
    s_tot = time.time()
    print('training...')
    
    best_ndcg = 0.
    best_metrics = {}
    
    # loss
    FLAG_mini_batch_CE = args.loss == 'mini_batch_CE'
    FLAG_full_CE = args.loss == 'full_CE'
    
    # timer
    t_data_enumeration = 0.
    t_data_transfer = 0.
    t_model_training = 0.
    
    e_batch_end = time.time()
    
    for epoch in range(max_epoch):
                
        for batch_id, batch in enumerate(train_dataloader):
            if not full_batch and batch_id >= batch_per_epoch:
                break
            
            s = time.time()
            
            model.train()
            idx_U, pos_idx_V, neg_idx_V = batch
            batch_size = idx_U.shape[0]
            
            # partial-structure mode for large graphs
            idx_U = idx_U.long().to(device)  # (batch, )
            pos_idx_V = pos_idx_V.long().to(device)  # (batch, )
            if FLAG_mini_batch_CE:
                if args.n_negs <= 0 or neg_idx_V is None:
                    raise ValueError('neg_idx_V is None!')
                neg_idx_V = neg_idx_V.long().to(device)  # (batch, n_negs)
            elif FLAG_full_CE:
                neg_idx_V = None
            else:
                raise ValueError(f'unsupported loss: {args.loss}')
            
            e_transfer = time.time()
            
            predictions, factors = model(idx_U, pos_idx_V, neg_idx_V)
            
            if FLAG_full_CE:
                truth = pos_idx_V
                l_fit = criterion(predictions, truth)
            elif FLAG_mini_batch_CE:
                truth = torch.zeros_like(pos_idx_V)
                l_fit = criterion(predictions, truth)
            else:
                raise ValueError(f'unsupported loss: {args.loss}')
                
            l_reg = regularizer(factors)
            
            batch_loss = l_fit + l_reg
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if batch_id % 100 == 0:
                print('training:', epoch, 'batch', batch_id, 'batch_loss:', batch_loss.item())
            
            e = time.time()
            
            t_data_enumeration += s - e_batch_end
            t_data_transfer += e_transfer - s
            t_model_training += e - e_transfer
            
            if args.test_batch_interval >= 0 and batch_id > 0 and batch_id % args.test_batch_interval == 0:
                print("epoch %d batch [ %d - %d ] model training time: %f s" % (epoch, batch_id - args.test_batch_interval, batch_id, t_model_training))
                print("epoch %d batch [ %d - %d ] data transfer time: %f s" % (epoch, batch_id - args.test_batch_interval, batch_id, t_data_transfer))
                print("epoch %d batch [ %d - %d ] data enumeration time: %f s" % (epoch, batch_id - args.test_batch_interval, batch_id, t_data_enumeration))
                print()
                
                # reset accumulated time
                t_model_training = 0.
                t_data_transfer = 0.
                t_data_enumeration = 0.
                
                FLAG_better_model = False
                
                if args.do_test:
                    print('-' * 20)
                    print('evaluating...')
                    s_eval = time.time()
                    metrics = batch_evaluation(args, model, csr_test, csr_train, epoch, device)
                    metrics['batch'] = batch_id
                    e_eval = time.time()
                    
                    if metrics[args.target_metric] >= best_ndcg:
                        best_metrics = metrics.copy()
                        best_ndcg = metrics[args.target_metric]
                        FLAG_better_model = True
                        
                    print('** epoch', epoch, 'batch', batch_id, '**')
                    print('Epoch', epoch, 'batch', batch_id, '|', end='\t')
                    print_metrics(args, metrics, args.print_max_K)
                    print('** best performance: epoch', best_metrics['epoch'], 'batch', best_metrics['batch'], '**')
                    print('Epoch', best_metrics['epoch'], 'batch', best_metrics['batch'], '|', end='\t')
                    print_metrics(args, best_metrics, args.print_max_K)

                    print()
                    print("evaluating time: %f s" % (e_eval - s_eval))
                    
                if (args.save_model and FLAG_better_model) or args.save_model_wo_test:
                    if not args.overwrite_model:
                        model_path = os.path.join(args.model_dir, 'model.epoch.%d.batch.%d.pt' % (epoch, batch_id))
                    else:
                        model_path = os.path.join(args.model_dir, 'model.pt')
                    torch.save(model.state_dict(), model_path)
                    if not args.overwrite_model:
                        print('save to', model_path)
                    else:
                        print('overwrite', model_path, 'epoch %d batch %d' % (epoch, batch_id))
                    
                print('-' * 20)
                print()
            
            # reset the ending time for each batch, to compute the time cost for data enumeration
            e_batch_end = time.time()
            
    e_tot = time.time()

    print("Total running time (training + evaluation): %f s" % (e_tot - s_tot))
