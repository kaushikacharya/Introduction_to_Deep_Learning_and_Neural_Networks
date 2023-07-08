import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union

## 1. DEFINE FeedForward, LayerNorm, SkipConnection
## 2. DEFINE ScaledDotProductAttention and MultiHeadAttention

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None,
                dropout: Optional[nn.Dropout] = None
                ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len)
            `dropout`: nn.Dropout
        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.sdpa = ScaledDotProductAttention()
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None
                ) -> torch.FloatTensor:
        """
        Args:
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
            `value`: shape (batch_size, max_len, d_model)
            `mask`: shape (batch_size, max_len)
        
        Returns:
            shape (batch_size, max_len, d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads. B*1*1*L
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        batch_size = query.size(0)

        # 1) Do all the linear projection in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
                              for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        # x: B * H * L * D_v
        x, self.attn = self.sdpa(query, key, value, mask, self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linears[-1](x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            `d_model`: model dimension
            `d_ff`: hidden dimension of feed forward layer
            `dropout`: dropout rate, default 0.1
        """
        super(FeedForward, self).__init__() 
        ## 1. DEFINE 2 LINEAR LAYERS AND DROPOUT HERE
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_ff)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)
        Returns:
            same shape as input x
        """
        ## 2.  RETURN THE FORWARD PASS 
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean)/(std + self.eps) + self.b

class SkipConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size: int, dropout: float) -> None:
        super(SkipConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,
                x: torch.FloatTensor,
                sublayer: Union[FeedForward, MultiHeadAttention]
                ) -> torch.FloatTensor:
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Encoder layer"""
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        ## 3. EncoderLayer subcomponents
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SkipConnection(size=size, dropout=dropout)) for _ in range(2)])
        self.size = size
    
    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        ## 4. EncoderLayer forward pass
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)

        return x

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer: EncoderLayer, N:  int):
        super(Encoder, self).__init__()
        ## 5. Encoder subcomponents
        self.encoder_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """"Pass the input (and mask) through each layer in turn."""
        ## 6. Encode forward pass
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        x = self.norm(x)
        
        return x

class TransformerEncoder(nn.Module):
    """The encoder of transformer
    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """
    def __init__(self, n_layers: int, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.encoder_layer = EncoderLayer(size=d_model, self_attn=self.multi_head_attention, feed_forward=self.feed_forward, dropout=dropout)
        self.encoder = Encoder(layer=self.encoder_layer, N=n_layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)

if __name__ == "__main__":
    # FeedForward(10, 100, 0.1)
    batch_size = 16
    d_model = 512
    d_ff = 512
    max_len = 10

    transformer_encoder = TransformerEncoder(n_layers=6, d_model=d_model, d_ff=d_ff, n_heads=8)
    x = torch.rand((batch_size, max_len, d_model))
    mask = torch.ones((batch_size, max_len))
    # In a batch, there would be sequences of varying length.
    # For a sequence with length < max_len, the remaining values are assigned zero to represent padding.
    vals = torch.randint(1, max_len, (batch_size,))

    for i in range(batch_size):
        mask[i, vals[i]+1:] = 0
    # Change data type to ByteTensor
    mask = mask.type(torch.ByteTensor)

    z = transformer_encoder(x, mask)
    print(f"z.shape: {z.shape}")
