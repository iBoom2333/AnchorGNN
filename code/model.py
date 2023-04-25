import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def get_U_emb(self, idx_U):
        raise NotImplementedError

    def get_V_emb(self):
        raise NotImplementedError


class BGE_Encoder(BaseModel):
    def __init__(self, dim, num_U, num_V, n_anchor, dim_anchor):
        super(BGE_Encoder, self).__init__()
        self.dim = dim
        self.num_U, self.num_V = num_U, num_V
        
        self.U_emb = nn.Embedding(num_U, dim)
        self.V_emb = nn.Embedding(num_V, dim)
        self.reset_parameters()

        self.anchor_conv = AnchorConv(dim, n_anchor, dim_anchor)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.U_emb.weight)
        nn.init.xavier_normal_(self.V_emb.weight)

    def forward(self, idx_U, pos_idx_V, neg_idx_V):
        lhs = self.U_emb(idx_U)
        rhs = self.V_emb(pos_idx_V)
        if neg_idx_V is not None:
            neg_rhs = self.V_emb(neg_idx_V)
        
        lhs = self.anchor_conv(lhs)
        
        if neg_idx_V is None:  # full CE
            to_score = self.V_emb.weight
            predictions = lhs @ to_score.transpose(0, 1)
        else:  # mini batch CE
            to_score = torch.cat((rhs.unsqueeze(dim=1), neg_rhs), dim=1)
            predictions = (lhs.unsqueeze(dim=1) * to_score).sum(-1)
            
        return (
            predictions
        ), (lhs, rhs)
    
    def get_U_emb(self, idx_U):
        lhs = self.U_emb(idx_U)
        lhs = self.anchor_conv(lhs)
        return lhs

    def get_V_emb(self):
        return self.V_emb.weight
    

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, input):
        return self.layer(input)


class MLP(nn.Module):
    def __init__(self, dims, act='relu', dropout=0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(FC(dims[i - 1], dims[i]))
        self.act = getattr(F, act)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        curr_input = input
        for i in range(len(self.layers) - 1):
            hidden = self.layers[i](curr_input)
            hidden = self.act(hidden)
            if self.dropout:
                hidden = self.dropout(hidden)
            curr_input = hidden
        output = self.layers[-1](curr_input)
        return output


class AnchorConv(nn.Module):
    def __init__(self, dim, n_anchor, dim_anchor):
        super(AnchorConv, self).__init__()
        self.dim = dim
        self.n_anchor = n_anchor
        self.dim_anchor = dim_anchor
        self.anchors = Anchors(self.n_anchor, self.dim_anchor)
        
        self.recv = MLP([self.n_anchor, self.dim])
        self.send = MLP([self.dim, dim_anchor])
        
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.n_anchor)
    
    def forward(self, input):
        input = self.norm1(input)
        s = self.send(input)
        r = self.anchors(s)
        r = self.norm2(r)
        a = self.recv(r)
        a = torch.sin(a)
        input = input + a
        return input


class Anchors(nn.Module):
    def __init__(self, n_anchor, dim_anchor):
        super(Anchors, self).__init__()
        self.anchor_emb = nn.Parameter(torch.empty(n_anchor, dim_anchor))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.anchor_emb)

    def forward(self, input):
        return input @ self.anchor_emb.transpose(0, 1)


class Regularizer(nn.Module):
    def __init__(self):
        super(Regularizer, self).__init__()
    
    def forward(self, tuples):
        raise NotImplementedError


class L2(Regularizer):
    def __init__(self, weight):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, tuples):
        norm = 0
        for f in tuples:
            norm += torch.norm(f, p=2)
        return self.weight * norm
    