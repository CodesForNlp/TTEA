import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # 反转边
    self_loop=torch.cat([edge_index_all[0].unique().unsqueeze(0),edge_index_all[0].unique().unsqueeze(0)],dim=0)
    # print(edge_index_all.size(0),edge_index_all.size(1),self_loop.size(0),self_loop.size(1))
    edge_index_all = torch.cat([edge_index_all, self_loop], dim=1)  # 反转边
    rel_all = torch.cat([rel, rel + rel.max() + 1])  # 反转关系
    return edge_index_all, rel_all

def get_train_batch(x1, x2, train_set, k=5):  # 获得负样本top5
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k + 1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k + 1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k + 1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k + 1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)

    # #trans产生最相似负样本
    # e1_negh = torch.cdist(x1[edge_index1[0][trans_index1]], x1, p=1).topk(k + 1, largest=False)[1].t()[1:]
    # e1_negt = torch.cdist(x1[edge_index1[1][trans_index1]], x1, p=1).topk(k + 1, largest=False)[1].t()[1:]
    # e2_negh = torch.cdist(x2[edge_index2[0][trans_index2]], x2, p=1).topk(k + 1, largest=False)[1].t()[1:]
    # e2_negt = torch.cdist(x2[edge_index2[1][trans_index2]], x2, p=1).topk(k + 1, largest=False)[1].t()[1:]
    # train_batch1 = torch.stack([e1_negh, e1_negt], dim=0)
    # train_batch2 = torch.stack([e2_negh, e2_negt], dim=0)
    return train_batch

def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print('Left:\t', end='')
    for k in Hn_nums:
        pred_topk = S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item() / pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk * 100), end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1 / (rank + 1)).mean().item()
    print('MRR: %.3f' % MRR)
    print('Right:\t', end='')
    for k in Hn_nums:
        pred_topk = S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item() / pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk * 100), end='')
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1 / (rank + 1)).mean().item()
    print('MRR: %.3f' % MRR)

def get_hits_stable(x1, x2, pair):
    pair_num = pair.size(0)
    S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    # index = S.flatten().argsort(descending=True)
    index = (S.softmax(1) + S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index // pair_num
    index_e2 = index % pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num * 100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned / pair_num * 100))

