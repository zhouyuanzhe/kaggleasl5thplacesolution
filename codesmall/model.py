import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import BertModel, BertConfig
from dataset import point_dim
import numpy as np

num_class  = 250
num_landmark = 543
max_length = 256
embed_dim  = 240
num_head   = 16

def positional_encoding(length, embed_dim):
    dim = embed_dim//2
    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)
    angle = 1 / (1000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)
    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


class XEmbed(nn.Module):
    def __init__(self,
    ):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(point_dim, embed_dim*2, bias=True),
            nn.LayerNorm(embed_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim*2, embed_dim, bias=True),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
    )
    def forward(self, x, x_mask):
        B,L,_ = x.shape
        v = self.v(x)
        x = v
        return x, x_mask

class TransformerBlock(nn.Module):
    def __init__(self,
        embed_dim,
        num_head,
        out_dim,
    ):
        super().__init__()
        self.attn  = MyMultiHeadAttention(
            embed_dim=embed_dim,
            out_dim=embed_dim,
            qk_dim=embed_dim // num_head,
            v_dim=embed_dim // num_head,
            num_head=num_head,
        )
        self.ffn   = FeedForward(embed_dim, out_dim*2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask=None):
        x = x + F.dropout(self.attn((self.norm1(x)), x_mask), 0.1)
        x = x + F.dropout(self.ffn((self.norm2(x))), 0.1)
        return x

class MyMultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            out_dim,
            qk_dim,
            v_dim,
            num_head,
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head  = num_head
        self.qk_dim = qk_dim
        self.v_dim  = v_dim

        self.q = nn.Linear(embed_dim, qk_dim*num_head)
        self.k = nn.Linear(embed_dim, qk_dim*num_head)
        self.v = nn.Linear(embed_dim, v_dim*num_head)

        self.out = nn.Linear(v_dim*num_head, out_dim)
        self.scale = 1/(qk_dim**0.5)

    #https://github.com/pytorch/pytorch/issues/40497
    def forward(self, x, x_mask):
        B,L,dim = x.shape
        #out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
        num_head = self.num_head
        qk_dim = self.qk_dim
        v_dim = self.v_dim

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(B, L, num_head, qk_dim).permute(0,2,1,3).contiguous()
        k = k.reshape(B, L, num_head, qk_dim).permute(0,2,3,1).contiguous()
        v = v.reshape(B, L, num_head, v_dim ).permute(0,2,1,3).contiguous()

        dot = torch.matmul(q, k) *self.scale  # H L L
        x_mask = x_mask.reshape(B,1,1,L).expand(-1,num_head,L,-1)
        #dot[x_mask]= -1e4
        dot.masked_fill_(x_mask, -1e4)
        attn = F.softmax(dot, -1)    # L L

        v = torch.matmul(attn, v)  # L H dim
        v = v.permute(0,2,1,3).reshape(B,L, v_dim*num_head).contiguous()
        out = self.out(v)

        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.mlp(x)

class M(nn.Module):

    def __init__(self, num_block):
        super().__init__()
        self.x_embed = XEmbed()

        pos_embed = positional_encoding(max_length, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed) 
        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))

        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_head,
                embed_dim,
            ) for i in range(num_block)
        ])
        self.logit = nn.Linear(embed_dim, num_class)

    def forward(self, x, dropout=None):
        x_mask = x.max(dim=2)[0].ne(0).long()

        x, x_mask = self.x_embed(x, x_mask)
        B,L,_ = x.shape
        if self.training:
            # randomly mask each element of the input sequence with probability 0.15
            B,L,_ = x.shape
            prob_mask = torch.empty(x_mask.shape[0], x_mask.shape[1]).uniform_(0, 1).to(device=x_mask.device)
            prob_mask = (prob_mask > 0.15).long()
            x_mask = x_mask * prob_mask
        x_mask = (x_mask<0.5)

        x = x + self.pos_embed[:L].unsqueeze(0)
        x = torch.cat([
            self.cls_embed.unsqueeze(0).repeat(B,1,1),
            x
        ],1)
        x_mask = torch.cat([
            torch.zeros(B,1).to(x_mask),
            x_mask
        ],1)
 
        for block in self.encoder:
            x = block(x,x_mask)

        if dropout:
            x = F.dropout(x,p=dropout)

        x_mask = x_mask.unsqueeze(-1)
        x_mask = 1-x_mask.float()
        last = (x*x_mask).sum(1)/x_mask.sum(1)
        logit = self.logit(last)
        return logit
