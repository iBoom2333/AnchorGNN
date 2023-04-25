import torch
import numpy as np
import random
import pickle


def set_random_seed(seed, set_cuda=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if set_cuda:
        torch.cuda.manual_seed(seed)
        
        
def print_metrics(args, metrics, print_max_K=True):
    if print_max_K:
        k = args.max_K
        print(f'precision@{k}:', metrics[f'precision@{k}'], end='\t')
        print(f'recall@{k}:', metrics[f'recall@{k}'], end='\t')
        print(f'ndcg@{k}:', metrics[f'ndcg@{k}'])
    else:
        for i, k in enumerate(args.topk):
            # if i > 0:
            #     print('--')
            print(f'precision@{k}:', metrics[f'precision@{k}'], end='\t')
            print(f'recall@{k}:', metrics[f'recall@{k}'], end='\t')
            print(f'ndcg@{k}:', metrics[f'ndcg@{k}'], end='\t')
            print()


def save_pickle(o, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)
