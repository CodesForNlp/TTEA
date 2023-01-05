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

def Semi_train(x1, x2, edge_index1,edge_index2,pairs,k):   #半监督训练过程
    device = torch.device('cuda:0')
    ids1 = set(torch.cat([edge_index1[0], edge_index1[1]], dim=0).unique().tolist())
    ids2 = set(torch.cat([edge_index2[0], edge_index2[1]], dim=0).unique().tolist())
    pair1 = set(pairs[0, :].tolist())
    pair2 = set(pairs[1, :].tolist())
    e1 = torch.tensor(list(ids1 - pair1)).to(device)      #获取不包含对齐种子的实体
    e2 = torch.tensor(list(ids2 - pair2)).to(device)

    # #新增margin
    # margin=torch.sum(torch.abs(x1[pairs[0, :]] - x2[pairs[1, :]]), dim=-1)
    # margin=torch.mean(margin.float(), dim=0)

    e12 = torch.cdist(x1[e1], x2[e2], p=1).topk(k, largest=False)[1].t()[0:]
    x=torch.cat([e1.unsqueeze(0),e12],dim=0)
    x=torch.stack([x[0],x[1]],dim=1).to(device)
    e21 = torch.cdist(x2[e2], x1[e1], p=1).topk(k, largest=False)[1].t()[0:]
    y=torch.cat([e21,e2.unsqueeze(0)],dim=0)
    y=torch.stack([y[0],y[1]],dim=1).to(device)
    # print(x.size(0),y.size(0))
    for i in torch.arange(e1.size(0)):
        # dis=torch.sum(torch.abs(x1[x[i][0]] - x2[x[i][1]]), dim =-1)
        if (x[i] in y):
            pair1=torch.cat([pairs[0,:], x[i][0].unsqueeze(0)], dim=0)
            pair2=torch.cat([pairs[1,:], x[i][1].unsqueeze(0)], dim=0)
            pairs=torch.stack([pair1, pair2], dim=0)
    return e1, e2, pairs

