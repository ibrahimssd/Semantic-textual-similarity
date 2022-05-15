###helpful resource: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch import Tensor
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self):
      super().__init__()
      self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
      # q, k, and v are batch-first
      # TODO: implement
      d = k.size(1)
      att_score = torch.matmul(q,torch.transpose(k,2,3))/math.sqrt(d)
      if mask is not None:
        mask = mask.unsqueeze(1)
        att_score = att_score.masked_fill(mask == 0, -1e11)
      att_score = self.dropout(F.softmax(att_score,dim=1))
      # print(att_score)
      z = torch.matmul(att_score,v)
      return z, att_score

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
      super().__init__()

      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.dim_r = self.embed_dim // self.num_heads   # to evenly split q, k, and v across heads.
      self.dotatt = Attention()

      self.q_linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
      self.k_linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
      self.v_linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
      self.final_linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
      
      # xavier initialization for linear layer weights
      nn.init.xavier_uniform_(self.q_linear_proj.weight)
      nn.init.xavier_uniform_(self.k_linear_proj.weight)
      nn.init.xavier_uniform_(self.v_linear_proj.weight)
      nn.init.xavier_uniform_(self.final_linear_proj.weight)

    def forward(self, q, k, v, mask=None):
      # q, k, and v are batch-first
      ########################################################################
      # TODO: Implement multi-head attention as described in Section 3.2.2
      # of the paper.        
      ########################################################################
      # shapes of q, k, v are [bsize, SEQ_LEN + 2, hidden_dim]
      bsize = k.shape[0]
      ##dividing the input matrix to heads(num_heads) of equal size(dim_r) and calculating the projections using the weights
      q_h = self.q_linear_proj(q).view(bsize,-1,self.num_heads, self.dim_r).transpose(1,2)
      k_h = self.k_linear_proj(k).view(bsize,-1,self.num_heads, self.dim_r).transpose(1,2)
      v_h = self.v_linear_proj(v).view(bsize,-1,self.num_heads, self.dim_r).transpose(1,2)
      ##calculating self attention for the heads
      # print(self.num_heads)
      z_h, att_score = self.dotatt(q_h,k_h,v_h, mask=mask)
      ##concatenating the heads outputs and projecting the output using the output weights
      z = z_h.transpose(1,2).reshape(bsize,-1,self.num_heads * self.dim_r) 
      mh_att = self.final_linear_proj(z)
      return mh_att, att_score

class Encoder(nn.Module):
    def __init__(self, num_hidden, num_heads):
      super().__init__()
      d_ff=2048 ##this value and feedforward structure is from paper section 3.3
      dropout = 0.1
      self.att = MultiHeadAttention(embed_dim=num_hidden, num_heads=num_heads)
      # TODO: add necessary member variables
      self.norm1 = nn.LayerNorm(num_hidden)
      self.norm2 = nn.LayerNorm(num_hidden)
      self.feedforward = nn.Sequential(
          nn.Linear(num_hidden,d_ff),
          nn.ReLU(),
          nn.Linear(d_ff,num_hidden),
      )
      self.dropout_1 = nn.Dropout(dropout)
      self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
      z, att = self.att(x, x, x, mask=mask)
      z = self.dropout_1(z)
      x = self.norm1(x+z) #add & norm 
      o = self.dropout_2(self.feedforward(x))
      o = self.norm2(x+o) #add & norm 
      return o, att

def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cuda:0"),
) -> Tensor:
      pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
      dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
      phase = pos / (1e4 ** (dim // dim_model))
      return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class TransformerEncoder(nn.Module):
    def __init__(self, N=1, num_hidden=128, num_heads=6, device="cuda:0"):
      super().__init__()
      d_ff = 2048
      self.N = N
      self.encoder = get_clones(Encoder(num_hidden, num_heads), N)
      self.fco = nn.Sequential(
          nn.Linear(num_hidden,d_ff),
          nn.ReLU(),
          nn.Linear(d_ff,num_hidden),
      )
      self.num_hidden = num_hidden
      self.device = device

    
    def forward(self, x, mask=None):
      # TODO: implement
      ##add positional encoding###
      seq_len, dimension = x.size(1), x.size(2)
      x += position_encoding(seq_len, dimension, self.device)
      ##encoder forward pass
      for i in range(self.N):
        z, att = self.encoder[i](x, mask=mask)
      return z, att