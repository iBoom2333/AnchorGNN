from tqdm import tqdm
import torch
import numpy as np


def precision_recall(r, k, n_ground_truth):
    right_pred = r[:, :k].sum(1)  # (batch, )
    n_ground_truth_denomitor = n_ground_truth.clone()
    n_ground_truth_denomitor[n_ground_truth_denomitor == 0] = 1
    batch_recall = (right_pred / n_ground_truth_denomitor).sum()
    batch_precision = right_pred.sum() / k
    return batch_recall, batch_precision


def ndcg(r, k, n_ground_truth):
    pred_data = r[:, :k]
    device = pred_data.device
    max_r = (torch.arange(k, device=device).expand_as(pred_data) < n_ground_truth.view(-1, 1)).float()  # (batch, k)
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2, device=device)), dim=1)  # (batch, ) as a denominator
    dcg = torch.sum(pred_data * (1. / torch.log2(torch.arange(2, k + 2, device=device))), dim=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    batch_ndcg = ndcg.sum()
    return batch_ndcg


def test_minibatch(csr_test, csr_train, test_batch):
    num_U = len(csr_test.indptr) - 1
    for begin in range(0, num_U, test_batch):
        head = csr_test.indptr[begin: min(begin + test_batch, num_U)]
        tail = csr_test.indptr[1 + begin: 1 + begin + test_batch]
        num_pos_V = tail - head
        # print('[', begin, begin + test_batch, ']', 'pos item cnt:', num_pos_V)
        # print('sum of n_items:', num_pos_V.sum())
        ground_truth = csr_test.indices[head[0]: tail[-1]]
        
        # assert num_pos_V.sum() == len(ground_truth)  # debug
        
        # print('data:', '(', len(ground_truth), ')', ground_truth)
        
        # exclude items in training set
        head_train = csr_train.indptr[begin: min(begin + test_batch, num_U)]
        tail_train = csr_train.indptr[1 + begin: 1 + begin + test_batch]
        num_V_to_exclude = tail_train - head_train
        V_to_exclude = csr_train.indices[head_train[0]: tail_train[-1]]
        
        # assert num_V_to_exclude.sum() == len(V_to_exclude)  # debug
        
        batch_size = len(num_pos_V)
        yield np.arange(begin, begin + batch_size), num_pos_V, ground_truth, num_V_to_exclude, V_to_exclude
    

def batch_evaluation(args, model, csr_test, csr_train, epoch, device):
    model.eval()
    
    V_emb = model.get_V_emb()
    max_K = args.max_K
    num_test_U = 0
    # device = model.device
    
    metrics = {}
    for k in args.topk:
        metrics[f'epoch'] = epoch
        metrics[f'precision@{k}'] = 0.
        metrics[f'recall@{k}'] = 0.
        metrics[f'ndcg@{k}'] = 0.
    
    with tqdm(total=csr_test.shape[0], desc=f'eval epoch {epoch}') as pbar:
        for i, batch in enumerate(test_minibatch(csr_test, csr_train, args.test_batch)):
            # print('-' * 20)
            # print('batch', i)
            idx_U, n_ground_truth, ground_truth, num_V_to_exclude, V_to_exclude = batch
            assert idx_U.shape == n_ground_truth.shape
            assert idx_U.shape == num_V_to_exclude.shape
            # print(idx_U.shape, n_ground_truth.shape, ground_truth.shape)

            batch_size = idx_U.shape[0]
            num_U_to_exclude = (n_ground_truth == 0).sum()  # exclude users that are not in test set
            # print('num_U_to_exclude:', num_U_to_exclude)
            num_test_U += batch_size - num_U_to_exclude
            
            # -> cuda 
            idx_U = torch.tensor(idx_U, dtype=torch.long, device=device)
            n_ground_truth = torch.tensor(n_ground_truth, dtype=torch.long, device=device)
            ground_truth = torch.tensor(ground_truth, dtype=torch.long, device=device)
            num_V_to_exclude = torch.tensor(num_V_to_exclude, dtype=torch.long, device=device)
            V_to_exclude = torch.tensor(V_to_exclude, dtype=torch.long, device=device)
            
            ########################################
            # metrics calculation
            
            with torch.no_grad():
                rating = model.get_U_emb(idx_U) @ V_emb.transpose(0, 1)  # (batch, num_V)
                row_index = torch.arange(batch_size, device=device)  # (batch, )
                
                # filter out the items in the training set
                row_index_to_exclude = row_index.repeat_interleave(num_V_to_exclude)
                rating[row_index_to_exclude, V_to_exclude] = -1e6
                
                # pick the top max_K items
                _, rating_K = torch.topk(rating, k=max_K)  # rating_K: (batch, max_K)
                
                # build a test_graph based on ground truth coordinates
                row_index_ground_truth = row_index.repeat_interleave(n_ground_truth)
                test_g = torch.sparse_coo_tensor(indices=torch.stack((row_index_ground_truth, ground_truth), dim=0), values=torch.ones_like(ground_truth), size=(batch_size, V_emb.size(0)))
                
                # build a pred_graph based on top max_K predictions
                pred_row = row_index.repeat_interleave(max_K)
                pred_col = rating_K.flatten()
                pred_g = torch.sparse_coo_tensor(indices=torch.stack((pred_row, pred_col), dim=0), values=torch.ones_like(pred_col), size=(batch_size, V_emb.size(0)))
                
                # build a hit_graph based on the intersection of test_graph and pred_graph
                dense_g = (test_g * pred_g).coalesce().to_dense().float()

                r = dense_g[pred_row, pred_col].view(batch_size, -1)  # (batch, max_K)
            
                # recall, precision, ndcg
                for k in args.topk:
                    # recall, precision
                    batch_recall, batch_precision = precision_recall(r, k, n_ground_truth)
                    # ndcg
                    batch_ndcg = ndcg(r, k, n_ground_truth)
                    
                    # print(f'batch_precision@{k}:', batch_precision.item())
                    # print(f'batch_recall@{k}:', batch_recall.item())
                    # print(f'batch_ndcg@{k}:', batch_ndcg.item())
                    # print('--')
                    
                    metrics[f'precision@{k}'] += batch_precision.item()
                    metrics[f'recall@{k}'] += batch_recall.item()
                    metrics[f'ndcg@{k}'] += batch_ndcg.item()
                    
            pbar.update(batch_size)
                
    for k in args.topk:
        metrics[f'precision@{k}'] /= num_test_U
        metrics[f'recall@{k}'] /= num_test_U
        metrics[f'ndcg@{k}'] /= num_test_U
            
    return metrics