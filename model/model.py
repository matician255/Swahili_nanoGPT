"""
This is the implementation of a GPT language model in this single file 
"""

import torch  # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F # type: ignore

#hyperparameters setup--------------------------------
batch_size = 64
block_size = 256 
n_embd = 384
n_heads = 6
n_layer = 6
eval_iters = 200
eval_interval = 100
learning_rate = 3e-4
max_iters = 5000
dropout = 0.2
vocab_size = len(chars) # type: ignore

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.num_heads = num_heads
    self.head_size = head_size
    self.heads_total_dimension = num_heads * head_size

    self.key = nn.Linear(n_embd, self.heads_total_dimension, bias=False) #-->(B, T, heads_total...)
    self.query = nn.Linear(n_embd, self.heads_total_dimension, bias=False)
    self.value = nn.Linear(n_embd, self.heads_total_dimension, bias=False)
    self.proj = nn.Linear(self.heads_total_dimension, n_embd)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2) #-->(B, nh, T, hs)
    q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2) #-->(B, nh, T, hs)
    v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2) #-->(B, nh, T, hs)

    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, nh, T, T)
    wei = F.softmax(wei, dim=-1) # (B, nh, T, T)
    wei = self.dropout(wei)

    # perform the weighted aggregation of the values
    out = wei @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    out = out.transpose(1, 2).contiguous().view(B, T, self.heads_total_dimension) #--> (B, T, h_t_d)
    out = self.dropout(self.proj(out))
    return out


class FeedFoward(nn.Module):
  """simple linear layer followed by a non linearity """

  def __init__(self, num_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """Transformer block : communication followed by computation"""

  def __init__(self,n_emb, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sa = MultiHeadAttention(n_heads, head_size)
    self.ffwd = FeedFoward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # type: ignore # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
          # crop idx to the last block_size tokens
          idx_cond = idx[:, -block_size:]
          # get the predictions
          logits, loss = self(idx_cond)
          # focus only on the last time step
          logits = logits[:, -1, :] # becomes (B, C)
          # apply softmax to get probabilities
          probs = F.softmax(logits, dim=-1) # (B, C)
          # sample from the distribution
          idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
          # append sampled index to the running sequence
          idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      return idx




