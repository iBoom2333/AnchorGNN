import argparse
import os
import sys
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
import random
from model import BGE_Encoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
from utils import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Generating Link Prediction Samples')
    parser.add_argument('--data_root_dir', default='./dataset', type=str, help='root path of input datasets')
    parser.add_argument('--output_root_dir', default='./dataset', type=str, help='root path of output LP samples')  # the same as input path by default
    parser.add_argument('--dataset', default='Pinterest', type=str)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--train_ratio', default=0.5, type=float, 
                        help='How many times is the number of training edges of LP compared to that of the testing edges?')
 
    args = parser.parse_args()
 
    args.input_dir = os.path.join(args.data_root_dir, args.dataset)
    args.output_dir = os.path.join(args.output_root_dir, args.dataset)
    
    os.makedirs(args.output_dir, exist_ok=True)
 
    return args


def specific_path(args, path):
    if path in ['train.csr.pickle', 'test.csr.pickle']:
        dirname = args.input_dir
    elif path in ['lp.train.npz', 'lp.test.npz']:
        dirname = args.output_dir
    else:
        raise ValueError('unsupported file!')
    return os.path.join(dirname, path)


def print_seperator():
    print('-' * 20)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    set_random_seed(args.seed, False)
    
    # load graphs: train, test
    print('> loading dataset')
    s = time.time()
    
    with open(specific_path(args, 'train.csr.pickle'), 'rb') as f:
        csr_train = pickle.load(f)
        print('train:', csr_train.shape, csr_train.nnz)
    with open(specific_path(args, 'test.csr.pickle'), 'rb') as f:
        csr_test = pickle.load(f)
        print('test:', csr_test.shape, csr_test.nnz)
    
    num_U, num_V = csr_train.shape
    train_src, train_dst = csr_train.nonzero()
    test_src, test_dst = csr_test.nonzero()
    train = np.stack((train_src, train_dst), axis=-1)
    test = np.stack((test_src, test_dst), axis=-1)
    
    e = time.time()
    print("loading time: %f s" % (e - s))
        
    # generate LP dataset
    print_seperator()
    print('> generating LP samples')
    s = time.time()
    
    n_test_pos = csr_test.nnz
    n_train_pos = int(args.train_ratio * n_test_pos)
        
    max_sample = (n_test_pos + n_train_pos) * 2
    print('max_sample:', max_sample)
    rand_row = torch.randint(0, num_U, (max_sample,))
    rand_col = torch.randint(0, num_V, (max_sample,))
    neg_g = torch.sparse_coo_tensor(indices=torch.stack((rand_row, rand_col), dim=0), values=torch.ones_like(rand_row), size=(num_U, num_V))
    print('initial neg_g:', neg_g.shape, neg_g._nnz())
    
    pos_g_edges = torch.cat((torch.from_numpy(train), torch.from_numpy(test)), dim=0).t()
    pos_g = torch.sparse_coo_tensor(indices=pos_g_edges, values=-torch.ones_like(pos_g_edges[0]), size=(num_U, num_V))
    print('pos_g:', pos_g.shape, pos_g._nnz())
    
    masked_g = (neg_g * pos_g).coalesce()
    neg_g = (neg_g + masked_g).coalesce()
    neg_indices = neg_g.indices()[:, neg_g.values() > 0]  # (2, n_neg)
    neg_indices = neg_indices.t()  # (n_neg, 2)
    if len(neg_indices) < n_test_pos + n_train_pos:
        raise ValueError('Generated negative samples are less then (test positive samples + train positive samples)!')
    e = time.time()
    print("generating time: %f s" % (e - s))
    
    # shuffle
    print_seperator()
    print('> shuffling LP training & testing samples')
    s = time.time()
    
    shuffled_pos_indices = np.random.permutation(len(train))
    shuffled_neg_indices = np.random.permutation(len(neg_indices))
    
    lp_pos_train = train[shuffled_pos_indices][:n_train_pos]
    lp_pos_test = test
    print('lp_pos_train:', lp_pos_train.shape)
    print('lp_pos_test:', lp_pos_test.shape)
    
    lp_neg = neg_indices[shuffled_neg_indices]
    lp_neg_train = lp_neg[:n_train_pos]
    lp_neg_test = lp_neg[n_train_pos: n_train_pos + n_test_pos]
    print('lp_neg:', lp_neg.shape)
    print('lp_neg_train:', lp_neg_train.shape)
    print('lp_neg_test:', lp_neg_test.shape)

    e = time.time()
    print("shuffling time: %f s" % (e - s))
    
    # save
    print_seperator()
    print('> saving file')
    np.savez_compressed(os.path.join(args.output_dir, 'lp.train.npz'), p=lp_pos_train, n=lp_neg_train)
    print('save to', os.path.join(args.output_dir, 'lp.train.npz'))
    np.savez_compressed(os.path.join(args.output_dir, 'lp.test.npz'), p=lp_pos_test, n=lp_neg_test)
    print('save to', os.path.join(args.output_dir, 'lp.test.npz'))
