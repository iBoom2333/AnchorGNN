import os
import pickle
from argparse import ArgumentParser
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from utils import save_pickle

def parse_args():
    parser = ArgumentParser("Generating Train/Test Splits for Link Prediction Datasets")
    parser.add_argument('--dataset', default='Pinterest')
    parser.add_argument('--input_file', default='edges.dat', type=str)  # preprocess GEBEp's "edges.dat" file by default
    parser.add_argument('--edge_root_dir', default='~/GEBE/data', type=str,
                        help='Root path of input files. Change it to your specified path.')
    parser.add_argument('--output_root_dir', default='./dataset', type=str, help='root path of output datasets')
    parser.add_argument('--test_ratio', default=0.4, type=float)  # Train/test ratio will be 6:4 by default.
    parser.add_argument('--seed', default=2022, type=int, help='seed')

    args = parser.parse_args()

    args.file = os.path.join(args.edge_root_dir, args.dataset, args.input_file)
    args.edge_dir = os.path.join(args.edge_root_dir, args.dataset)
    args.output_dir = os.path.join(args.output_root_dir, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def print_seperator():
    print('-' * 20)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    np.random.seed(args.seed)
    
    cnt_U = Counter()
    cnt_V = Counter()
    
    print('> reading files')
    with open(args.file) as f:
        for line in f:
            u, v, weight = line.split()
            cnt_U[u] += 1
            cnt_V[v] += 1

    print_seperator()
    print('> mapping nodes')
    u_mapping = dict((k, i) for i, k in enumerate(cnt_U.keys()))
    v_mapping = dict((k, i) for i, k in enumerate(cnt_V.keys()))
    
    num_U = len(u_mapping)
    num_V = len(v_mapping)
    print('num_U:', num_U)
    print('num_V:', num_V)
    
    save_pickle(u_mapping, os.path.join(args.output_dir, 'u_mapping.pickle'))
    save_pickle(v_mapping, os.path.join(args.output_dir, 'v_mapping.pickle'))

    ####################################################
    
    print_seperator()
    print('> reading valid edges')
    
    src = []
    dst = []
    w = []
    
    with open(args.file) as f:
        for line in f:
            u, v, weight = line.split()
            if u in u_mapping and v in v_mapping:
                u_id = u_mapping[u]
                v_id = v_mapping[v]
                weight = float(weight)
                src.append(u_id)
                dst.append(v_id)
                w.append(weight)
    
    src = np.array(src)
    dst = np.array(dst)
    w = np.array(w)
    
    # split training and test sets
    print_seperator()
    print('> train_test_split')
    edges = np.stack((src, dst), axis=-1)
    train, test = train_test_split(edges, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
    print('train edges:', train.shape)
    # print('initial test edges:', test.shape)
    print('test edges:', test.shape)
    
    csr_train = csr_matrix((np.ones(train.shape[0]), (train[:, 0], train[:, 1])), shape=(num_U, num_V))
    csr_test = csr_matrix((np.ones(test.shape[0]), (test[:, 0], test[:, 1])), shape=(num_U, num_V))
    
    print('train graph:', csr_train.shape, csr_train.nnz)
    print('test graph:', csr_test.shape, csr_test.nnz)
    
    # save files
    print_seperator()
    print('> saving file')
    save_pickle(csr_train, os.path.join(args.output_dir, 'train.csr.pickle'))
    print('save to', os.path.join(args.output_dir, 'train.csr.pickle'))
    save_pickle(csr_test, os.path.join(args.output_dir, 'test.csr.pickle'))
    print('save to', os.path.join(args.output_dir, 'test.csr.pickle'))
